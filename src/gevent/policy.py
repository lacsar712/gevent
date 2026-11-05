# Copyright (c) 2026 gevent contributors. See LICENSE for details.
"""
Per-greenlet dispatch/join runtime policy stacks.

This module provides a *policy stack* that is bound to the currently
running cooperative scope (the current greenlet).  Each stack frame
temporarily overrides three phases of task lifecycle handling:

1. **spawn_admit** -- what happens when this scope dispatches a new
   unit of work through ``gevent.spawn`` / ``Greenlet.spawn`` /
   ``spawn_later``, ``Group.spawn`` / ``Pool.spawn`` and
   ``ThreadPool.spawn``:

   * :data:`HONOR_CFG` -- the existing semantics apply unchanged,
     including a full :class:`~gevent.pool.Pool` blocking (or raising
     :class:`~gevent.pool.PoolFull`) and a full
     :class:`~gevent.threadpool.ThreadPool` blocking.
   * :data:`DENY_NEW` -- every new dispatch exit reached while this
     frame is on top raises :class:`SpawnDenied`.  Greenlets already
     running are never killed by this gate.
   * :data:`ONE_MORE` -- exactly one more dispatch succeeds while the
     frame is on top; the next and all further dispatches raise
     :class:`SpawnDenied`.

2. **join_gate** -- how rendezvous waits (``Greenlet.join/get``,
   ``joinall``, ``Group.join``) and ``Timeout``/``with_timeout`` behave:

   * :data:`HONOR_CFG` -- the configured *seconds* arguments apply
     unchanged.
   * :data:`FREEZE` -- frame-related waits cannot be cut short into a
     failure by a join timeout or a :class:`~gevent.timeout.Timeout`;
     already started children are not half-killed with a faked
     successful return.  Pending timeouts are suspended (recalculated)
     and resumed when the frame pops.
   * :data:`SHRINK_NOW` -- the effective wait bound becomes
     ``min(configured seconds, frame seconds)``; expirations still
     surface through the existing timeout/join failure surfaces.

3. **exit_reap** -- what happens to the units of work successfully
   dispatched *by this frame* when the frame pops, normally or due to
   an exception:

   * :data:`HONOR_CFG` -- no extra cleanup.
   * :data:`LEAVE_RUNNING` -- dispatched greenlets keep running; this
     never touches error propagation or ``link_exception`` callbacks.
   * :data:`FORCE_KILL` -- every still-unfinished unit this frame
     successfully dispatched is killed to an observable termination.
     Units dispatched by sibling/outer scopes are never touched.

The stack is per-greenlet: pushing on one greenlet does not change
what any other greenlet on the same hub observes, and two greenlets
pushing different frames never share state.  Child greenlets start
with their own empty stack; nested scopes always read the *current*
top, never a snapshot captured when an outer scope was entered.

Stacks never modify ``Pool.size``, ``ThreadPool.maxsize`` or any
``Timeout`` default.  No state is hung off ``Greenlet``/``Pool``/
``Hub`` instances: all bookkeeping lives in this module.
"""
from __future__ import absolute_import, division, print_function

import time
from contextlib import contextmanager
from weakref import WeakKeyDictionary

from greenlet import getcurrent

__all__ = [
    # spawn_admit values
    'HONOR_CFG',
    'DENY_NEW',
    'ONE_MORE',
    # join_gate values
    'FREEZE',
    'SHRINK_NOW',
    # exit_reap values
    'LEAVE_RUNNING',
    'FORCE_KILL',
    # API
    'PolicyFrame',
    'push',
    'pop',
    'scope',
    'current',
    'frames',
    'can_spawn_now',
    'join_timeout',
    'unguarded',
    # exceptions
    'SpawnDenied',
    'PolicyError',
    'TaskReaped',
]

#: spawn_admit / join_gate / exit_reap "do nothing different" value.
HONOR_CFG = 'honor_cfg'

#: spawn_admit: reject all new dispatches while the frame is on top.
DENY_NEW = 'deny_new'

#: spawn_admit: allow exactly one further successful dispatch.
ONE_MORE = 'one_more'

#: join_gate: frame-related waits are not interrupted by timeouts.
FREEZE = 'freeze'

#: join_gate: tighten the effective wait to a frame-provided bound.
SHRINK_NOW = 'shrink_now'

#: exit_reap: leave everything this frame dispatched running.
LEAVE_RUNNING = 'leave_running'

#: exit_reap: kill everything still unfinished that this frame dispatched.
FORCE_KILL = 'force_kill'

_VALID_SPAWN_ADMIT = frozenset((HONOR_CFG, DENY_NEW, ONE_MORE))
_VALID_JOIN_GATE = frozenset((HONOR_CFG, FREEZE, SHRINK_NOW))
_VALID_EXIT_REAP = frozenset((HONOR_CFG, LEAVE_RUNNING, FORCE_KILL))


def _monotonic():
    # Indirection so tests/embedders can see a single clock source.
    return time.monotonic()


class SpawnDenied(RuntimeError):
    """
    Raised by a dispatch exit (``spawn``/``spawn_later``/
    ``Pool.spawn``/``ThreadPool.spawn`` ...) when the current policy
    frame refuses new dispatches (``deny_new`` or an exhausted
    ``one_more``).
    """


class PolicyError(RuntimeError):
    """
    Raised for misuse of the policy machinery: invalid arguments,
    non-LIFO :func:`pop`, or pushing a frame in a state where the
    requested rendezvous change cannot be applied.
    """


class TaskReaped(RuntimeError):
    """
    Set on a :class:`~gevent.event.AsyncResult` returned by
    :meth:`ThreadPool.spawn <gevent.threadpool.ThreadPool.spawn>` when
    a ``force_kill`` frame pops while the task is still queued (never
    started in a worker thread).  Running native-thread tasks cannot
    be forcibly stopped.
    """


# Per-greenlet state.  This registry (and never an attribute on
# Greenlet/Pool/Hub) is the whole policy storage.
#
# value layout:
#   'stack'    : list[PolicyFrame], index -1 is the top
#   'neutral'  : int, >0 while inside unguarded() internal bookkeeping
#   'timeouts' : set[Timeout], timeouts currently owned by the greenlet
_states = WeakKeyDictionary()

# Timeout -> {'deadline': float, 'target': greenlet, 'suspended': bool}
_timeout_meta = WeakKeyDictionary()


def _get_state(create=False):
    current = getcurrent()
    state = _states.get(current)
    if state is None and create:
        state = {
            'stack': [],
            'neutral': 0,
            'timeouts': set(),
            'waits': [],
        }
        _states[current] = state
    return state


def _top(state=None):
    if state is None:
        state = _get_state()
    if state is not None:
        stack = state['stack']
        if stack:
            return stack[-1]
    return None


class _GreenletReapHandle(object):
    __slots__ = ('greenlet',)

    def __init__(self, greenlet):
        self.greenlet = greenlet

    def reap(self):
        glet = self.greenlet
        if getattr(glet, 'ready', None) is None or not glet.ready():
            # ``kill`` on an unstarted (e.g. spawn_later pending)
            # greenlet cancels the start and fires the existing links;
            # it never raises and never touches greenlets this frame
            # did not dispatch.
            kill = getattr(glet, 'kill', None)
            if kill is not None:
                kill()


class _ThreadTaskReapHandle(object):
    __slots__ = ('pool', 'result', 'task', 'thread_result', 'semaphore')

    def __init__(self, pool, result, task, thread_result, semaphore):
        self.pool = pool
        self.result = result
        self.task = task
        self.thread_result = thread_result
        self.semaphore = semaphore

    def reap(self):
        result = self.result
        if result.ready():
            return
        queue = self.pool.task_queue
        removed = False
        # Try to take the still-queued task back before a worker runs
        # it.  This runs on the hub-owning thread; the queue lock plus
        # the GIL make the dequeue decision authoritative.
        try:
            with queue._mutex:
                deque = queue._queue
                if self.task in deque:
                    deque.remove(self.task)
                    queue.unfinished_tasks -= 1
                    removed = True
        except Exception: # pylint:disable=broad-except
            removed = False

        if not removed:
            # Either running in a native thread (cannot be killed) or
            # already consumed; leave the result to the worker.
            return

        # Balance the slot spawn() acquired and tear down the cross-
        # thread watcher exactly as ThreadResult._on_async would have.
        try:
            self.thread_result.destroy_in_main_thread()
        except Exception: # pylint:disable=broad-except
            pass
        try:
            self.semaphore.release()
        except Exception: # pylint:disable=broad-except
            pass
        result.set_exception(
            TaskReaped("task removed by force_kill policy frame"))


class PolicyFrame(object):
    """
    One entry of the per-greenlet policy stack.

    Instances are produced by :func:`push` (and :func:`scope`); they
    are not intended to be constructed directly.
    """

    __slots__ = (
        'spawn_admit',
        'join_gate',
        'exit_reap',
        'seconds',
        '_deadline',
        '_spawned',
        '_one_used',
        '_records',
        '_closed',
    )

    def __init__(self, spawn_admit, join_gate, exit_reap, seconds):
        self.spawn_admit = spawn_admit
        self.join_gate = join_gate
        self.exit_reap = exit_reap
        self.seconds = seconds
        self._deadline = None if seconds is None else _monotonic() + seconds
        # Reap handles for units successfully dispatched while this
        # frame is on top.
        self._spawned = []
        self._one_used = False
        # [Timeout, deadline-before-frame, was-armed-before-frame]
        self._records = []
        self._closed = False

    def __repr__(self):
        return ('<PolicyFrame spawn_admit=%r join_gate=%r exit_reap=%r'
                ' seconds=%r>' % (
                    self.spawn_admit, self.join_gate, self.exit_reap,
                    self.seconds))

    @property
    def remaining(self):
        """
        Seconds until this frame's ``shrink_now`` deadline elapses.
        ``None`` for frames that carry no deadline.
        """
        if self._deadline is None:
            return None
        return self._deadline - _monotonic()


# ----------------------------------------------------------------------------
# Push / pop
# ----------------------------------------------------------------------------

def _validate(spawn_admit, join_gate, exit_reap, seconds):
    if spawn_admit not in _VALID_SPAWN_ADMIT:
        raise PolicyError(
            "Invalid spawn_admit %r; expected one of %r" % (
                spawn_admit, sorted(_VALID_SPAWN_ADMIT)))
    if join_gate not in _VALID_JOIN_GATE:
        raise PolicyError(
            "Invalid join_gate %r; expected one of %r" % (
                join_gate, sorted(_VALID_JOIN_GATE)))
    if exit_reap not in _VALID_EXIT_REAP:
        raise PolicyError(
            "Invalid exit_reap %r; expected one of %r" % (
                exit_reap, sorted(_VALID_EXIT_REAP)))
    if join_gate == SHRINK_NOW:
        if seconds is None:
            raise PolicyError(
                "join_gate=shrink_now requires a seconds value")
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise PolicyError(
                "shrink_now seconds must be a non-negative number: %r"
                % (seconds,))
    elif seconds is not None and join_gate != SHRINK_NOW:
        # Accepted but ignored elsewhere; be explicit instead of
        # silently surprising callers.
        raise PolicyError(
            "seconds is only meaningful with join_gate=shrink_now")


def _adopt_timeouts(frame, state):
    """
    Recalculate timeouts the greenlet already owns for the newly
    pushed frame (freeze -> suspend, shrink -> tighten).
    """
    now = _monotonic()
    for timeout in list(state['timeouts']):
        meta = _timeout_meta.get(timeout)
        if meta is None:
            continue
        suspended = meta['suspended']
        armed = not suspended and timeout.pending
        if not armed and not suspended:
            # Fired or cancelled outside of policy management.
            continue
        frame._records.append([timeout, meta['deadline'], armed])
        if frame.join_gate == FREEZE:
            if armed:
                from gevent import timeout as _timeout_mod
                _timeout_mod._suspend_timeout(timeout)
                meta['suspended'] = True
        elif frame.join_gate == SHRINK_NOW:
            bound = frame._deadline
            if bound < meta['deadline']:
                from gevent import timeout as _timeout_mod
                _timeout_mod._rearm_timeout(
                    timeout, max(0.0, bound - now),
                    meta['target'])
                meta['deadline'] = bound
                meta['suspended'] = False


def push(spawn_admit=HONOR_CFG,
         join_gate=HONOR_CFG,
         exit_reap=HONOR_CFG,
         seconds=None):
    """
    Push a :class:`PolicyFrame` onto the *current* greenlet's stack.

    The returned frame must be removed with :func:`pop` (LIFO), or
    used through the :func:`scope` context manager, which also pops on
    exceptions.
    """
    _validate(spawn_admit, join_gate, exit_reap, seconds)
    state = _get_state(create=True)
    if state['waits']:
        # A join/get rendezvous started under the frame that is still
        # on this greenlet cannot be retroactively re-governed; reject
        # instead of silently keeping the stale view.
        raise PolicyError(
            "cannot push a join_gate frame while a rendezvous wait is"
            " in progress")
    frame = PolicyFrame(spawn_admit, join_gate, exit_reap, seconds)
    if join_gate != HONOR_CFG and not state['neutral']:
        _adopt_timeouts(frame, state)
    state['stack'].append(frame)
    return frame


def _outer_view(stack):
    """
    Merge the join_gate view of the frames remaining after the popping
    frame has been removed.

    Returns ('suspend', None) or ('arm', absolute deadline or None).
    """
    bound = None
    for frame in reversed(stack):
        if frame.join_gate == FREEZE:
            return 'suspend', None
        if frame.join_gate == SHRINK_NOW:
            if bound is None or frame._deadline < bound:
                bound = frame._deadline
    return 'arm', bound


def _restore_timeouts(frame, stack):
    if not frame._records:
        return
    from gevent import timeout as _timeout_mod
    fake = _timeout_mod._FakeTimer
    now = _monotonic()
    mode, bound = _outer_view(stack)
    for record in frame._records:
        timeout, prev_deadline, _prev_armed = record
        meta = _timeout_meta.get(timeout)
        if meta is None or timeout.timer is fake:
            # Closed/cancelled while the frame was active.
            continue
        target_deadline = prev_deadline if bound is None else min(
            prev_deadline, bound)
        if mode == 'suspend':
            if not meta['suspended'] and timeout.pending:
                _timeout_mod._suspend_timeout(timeout)
                meta['suspended'] = True
            meta['deadline'] = target_deadline
            continue

        if meta['suspended']:
            # Frozen by this frame (or an adopted one this frame
            # covered); resume under the outer view.
            meta['suspended'] = False
            meta['deadline'] = target_deadline
            _timeout_mod._rearm_timeout(
                timeout, max(0.0, target_deadline - now), meta['target'])
        elif timeout.pending:
            if target_deadline < meta['deadline']:
                meta['deadline'] = target_deadline
                _timeout_mod._rearm_timeout(
                    timeout, max(0.0, target_deadline - now),
                    meta['target'])
        # else: already fired -- leave the fired surface untouched.


def _reap(frame):
    if frame.exit_reap != FORCE_KILL:
        return
    for handle in tuple(frame._spawned):
        try:
            handle.reap()
        except Exception: # pylint:disable=broad-except
            # Reaping must never mask the body's exception (or turn a
            # normal pop into a failure); kill() itself never raises.
            pass


def pop(frame=None):
    """
    Remove *frame* (default: the top frame) from the current greenlet.

    LIFO order is enforced.  Pending timeouts are restored to the
    outer view; ``force_kill`` frames reap their dispatched set on the
    way out, whether or not the scope body raised.
    """
    state = _get_state()
    if state is None or not state['stack']:
        raise PolicyError("no policy frame to pop on this greenlet")
    stack = state['stack']
    top = stack[-1]
    if frame is not None and frame is not top:
        raise PolicyError(
            "non-LIFO pop: %r is not the top frame %r" % (frame, top))
    stack.pop()
    top._closed = True
    try:
        _restore_timeouts(top, stack)
    finally:
        _reap(top)
        top._spawned = []
        top._records = []
    return top


@contextmanager
def scope(spawn_admit=HONOR_CFG,
          join_gate=HONOR_CFG,
          exit_reap=HONOR_CFG,
          seconds=None):
    """
    Context manager for :func:`push`/:func:`pop`.

    The frame is always popped (and ``force_kill`` reaping always runs)
    even if the suite raises.
    """
    frame = push(spawn_admit, join_gate, exit_reap, seconds)
    try:
        yield frame
    finally:
        pop(frame)


def current():
    """
    Return the current top :class:`PolicyFrame`, or ``None`` when the
    stack is empty (plain gevent API semantics).
    """
    return _top()


def frames():
    """
    Return a tuple with a snapshot of the current stack, outermost
    first.  Always re-read the live stack; never cache it across
    dispatches.
    """
    state = _get_state()
    if state is None:
        return ()
    return tuple(state['stack'])


# ----------------------------------------------------------------------------
# spawn_admit hooks (called from greenlet.py / pool.py / threadpool.py)
# ----------------------------------------------------------------------------

def _gate_frame(state):
    if state is None or state['neutral']:
        return None
    return _top(state)


def can_spawn_now():
    """
    Observation helper: whether the next dispatch on *this* greenlet
    would be admitted by the current top frame (ignoring pool capacity
    and other configured limits, which are still enforced separately
    under ``honor_cfg``).
    """
    frame = _gate_frame(_get_state())
    if frame is None or frame.spawn_admit == HONOR_CFG:
        return True
    if frame.spawn_admit == DENY_NEW:
        return False
    # ONE_MORE
    return not frame._one_used


def spawn_precheck():
    """
    Non-reserving gate, evaluated before any capacity wait.

    Raises :class:`SpawnDenied` if the current top frame would refuse
    the next dispatch.  Never consumes the ``one_more`` slot.
    """
    frame = _gate_frame(_get_state())
    if frame is None:
        return
    if frame.spawn_admit == DENY_NEW:
        raise SpawnDenied(
            "new task dispatch denied by deny_new policy frame %r"
            % (frame,))
    if frame.spawn_admit == ONE_MORE and frame._one_used:
        raise SpawnDenied(
            "new task dispatch denied: one_more policy frame %r already"
            " admitted one dispatch" % (frame,))


def _commit(frame):
    if frame.spawn_admit == DENY_NEW:
        raise SpawnDenied(
            "new task dispatch denied by deny_new policy frame %r"
            % (frame,))
    if frame.spawn_admit == ONE_MORE:
        if frame._one_used:
            raise SpawnDenied(
                "new task dispatch denied: one_more policy frame %r"
                " already admitted one dispatch" % (frame,))
        frame._one_used = True


def spawn_commit_greenlet(greenlet):
    """
    Reserve the dispatch and register the successfully-started
    greenlet on the current top frame.  Reads the live stack top.
    """
    frame = _gate_frame(_get_state())
    if frame is None:
        return
    _commit(frame)
    frame._spawned.append(_GreenletReapHandle(greenlet))


def spawn_commit_thread_task(pool, result, task, thread_result, semaphore):
    """
    Reserve the dispatch and register a successfully enqueued
    ThreadPool task on the current top frame.
    """
    frame = _gate_frame(_get_state())
    if frame is None:
        return
    _commit(frame)
    frame._spawned.append(
        _ThreadTaskReapHandle(pool, result, task, thread_result, semaphore))


@contextmanager
def unguarded():
    """
    Context manager marking gevent's own bookkeeping dispatches and
    waits (threadpool manager greenlets, resolver lookups, server
    shutdown helpers, ...) as infrastructure actions.

    Such actions are invisible to every policy gate: they are not
    refused, not registered for reaping, and do not consume an
    ``one_more`` slot.
    """
    state = _get_state(create=True)
    state['neutral'] += 1
    try:
        yield
    finally:
        state['neutral'] -= 1


# ----------------------------------------------------------------------------
# join_gate hooks (called from greenlet.py / pool.py / timeout.py)
# ----------------------------------------------------------------------------

def _effective_timeout(frame, timeout):
    if frame is None or frame.join_gate == HONOR_CFG:
        return timeout
    if frame.join_gate == FREEZE:
        return None
    # SHRINK_NOW
    remaining = frame._deadline - _monotonic()
    if remaining < 0:
        remaining = 0.0
    if timeout is None:
        return remaining
    return min(timeout, remaining)


def join_timeout(timeout=None):
    """
    Observation helper: the effective rendezvous timeout after merging
    *timeout* with the current top frame's ``join_gate``.
    """
    frame = _gate_frame(_get_state())
    return _effective_timeout(frame, timeout)


def join_begin(timeout=None):
    """
    Open a rendezvous wait (``join/get``/``joinall``/``Group.join``).

    Returns the timeout to actually wait with: ``None`` under
    ``freeze``, the merged bound under ``shrink_now``, otherwise the
    argument unchanged.

    Waits started with an empty stack are untracked (a greenlet that
    is parked cannot push a frame); waits started under a frame are
    marked, so a conflicting push after the rendezvous started can be
    explicitly rejected instead of silently keeping the stale view.
    """
    state = _get_state()
    frame = _gate_frame(state)
    if frame is None:
        return timeout
    state['waits'].append(frame)
    return _effective_timeout(frame, timeout)


def join_end():
    """Close a rendezvous wait opened by :func:`join_begin`."""
    state = _get_state()
    if state is not None and state['waits']:
        state['waits'].pop()


# ----------------------------------------------------------------------------
# Timeout hooks (called from timeout.py)
# ----------------------------------------------------------------------------

def timeout_starting(timeout, seconds):
    """
    Called by :meth:`Timeout.start <gevent.timeout.Timeout.start>` when
    *seconds* is not ``None``.

    Registers the timeout with the dispatching greenlet and merges it
    with the current top frame.

    :return: the seconds to arm the native timer with, or ``None`` when
             the frame suspends the timeout (``freeze``).
    """
    if seconds is None:
        return None
    state = _get_state(create=True)
    target = getcurrent()
    now = _monotonic()
    deadline = now + seconds
    meta = {'deadline': deadline, 'target': target, 'suspended': False}
    _timeout_meta[timeout] = meta
    state['timeouts'].add(timeout)

    frame = _gate_frame(state)
    if frame is None or frame.join_gate == HONOR_CFG:
        return seconds

    # Recorded so frame pop restores the pre-frame (configured)
    # deadline even for a timeout created inside the frame.
    frame._records.append([timeout, deadline, False])

    if frame.join_gate == FREEZE:
        meta['suspended'] = True
        return None

    # SHRINK_NOW
    remaining = frame._deadline - now
    if remaining < 0:
        remaining = 0.0
    if remaining < seconds:
        meta['deadline'] = now + remaining
        return remaining
    return seconds


def timeout_stopping(timeout):
    """
    Called by ``Timeout.cancel``/``close``: drop all policy bookkeeping
    for the timeout.
    """
    state = _get_state()
    if state is not None:
        state['timeouts'].discard(timeout)
    _timeout_meta.pop(timeout, None)
