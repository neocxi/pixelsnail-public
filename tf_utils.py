from __future__ import division
from __future__ import print_function

from cached_property import cached_property
import pickle
import functools
import inspect
import numpy as np
import os
import re
import threading
import time

import tensorflow as tf
from tensorflow.python.framework.dtypes import _NP_TO_TF
from tensorflow.python.util import nest

NP_TO_TF = {np_type: tf_type for np_type, tf_type in _NP_TO_TF}


class Struct(dict):
  """A dict that exposes its entries as attributes."""

  def __init__(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs)
    self.__dict__ = self

  @staticmethod
  def make(obj):
    """Modify `obj` by replacing `dict`s with `tfu.Struct`s."""
    if isinstance(obj, dict):
      new_obj = type(obj)() if isinstance(obj, Struct) else Struct()
      for k, v in obj.items():
        new_obj[k] = Struct.make(v)
      obj = new_obj

    elif nest.is_sequence(obj):
      return type(obj)(Struct.make(v) for v in obj)

    return obj


def _map_until(func, nested, until=None):
  if until is None:
    until = nest.not_seq_or_dict

  if until(nested):
    return func(nested)

  elif isinstance(nested, dict):
    output = Struct()
    for k, v in nested.items():
      output[k] = _map_until(func, v, until)
    return output

  elif nest.is_sequence(nested):
    return tuple(_map_until(func, v, until) for v in nested)

  else:
    raise ValueError


nest.map = _map_until


def _flatten_dict(d):
  keys, flat_vals = sorted(d.keys()), []
  for k in keys:
    flat_vals.extend(nest.flatten(d[k]))
  return flat_vals


def _pack_dict(dct, lst):
  d, keys = Struct(), sorted(dct)
  for k in keys:
    n = len(nest.flatten(dct[k]))
    d[k] = nest.pack_sequence_as(dct[k], lst[:n])
    lst = lst[n:]
  return d


def shape_if_known(tensor, dim):
  assert isinstance(dim, int)
  val = tensor.get_shape()[dim].value
  if val is None:
    val = tf.shape(tensor)[dim]
  return val


def shape(tensor):
  return [shape_if_known(tensor, i) for i in range(tensor.shape.ndims)]


def concat_shapes(*shapes):
  shape = tf.TensorShape([])
  for s in shapes:
    s = [None] if s is None else s
    shape = shape.concatenate(tf.TensorShape(s))
  return shape


def _list_flatten(structure):
  out = []
  if isinstance(structure, list):
    for s in structure:
      out.extend(_list_flatten(s))
  else:
    out = [structure]
  return out


def Session(devices=None, frac=None, **config):
  """Create a session."""
  if devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

  if frac is not None:
    config['gpu_options'] = tf.GPUOptions(per_process_gpu_memory_fraction=frac)

  return tf.Session(config=tf.ConfigProto(**config))


def vdef(*shape, dtype=tf.float32):
  shape = nest.flatten(shape)
  if len(shape) > 0:
    if isinstance(shape[-1], tf.DType) or shape[-1] in NP_TO_TF or isinstance(shape[-1], np.dtype):
      dtype = shape.pop(-1)
  return Struct(shape=concat_shapes(*shape), dtype=dtype)


def make_placeholders(variables):
  """Shortcut to make placholders.

  Args:
    variables: A dict where the keys are str and values are nested structures of (tfu.Struct, tf.TensorShape).

  Returns:
    A dict(name: tf.placeholder) with the same keys as `variables`.
  """
  placeholders = Struct()
  for name, args in variables.items():
    def _make(pl):
      if isinstance(pl, tf.TensorShape):
        return tf.placeholder(tf.float32, pl, name)
      else:
        return tf.placeholder(pl.dtype, pl.shape, name)

    placeholders[name] = nest.map(_make, args,
                                  until=lambda x: any(isinstance(x, t) for t in {dict, Struct, tf.TensorShape}))

  return placeholders


def pl_like_tensor(tensor, name=None):
  return tf.placeholder(tensor.dtype, tensor.get_shape().as_list(),
                        name=name)


def placeholders_like(tensors):
  return nest.map(pl_like_tensor, tensors)


class Function(object):

  def __init__(self, inputs, outputs,
               session=tf.get_default_session, name='function'):
    """Create a function interface to `session.run()`.

    Args:
      inputs: A dict, keys are strings and values are nested structures that can be values in a `feed_dict`.
        Leaves are placeholders, and will be replaced by numpy arrays when the function is evaluated.
      outputs: Any nested structure that can be evaluated by `session.run()`.
      session: A callable that returns a `tf.Session`.
      name: A string.
    """
    self.session, self.name = session, name
    self.inputs, self.outputs = inputs, outputs

  def __call__(self, **values):
    session = self.session()
    feed = {pl: values[name]
            for name, pl in self.inputs.items()}
    result = session.run(self.outputs, feed_dict=feed)
    return Struct.make(result)

  def __str__(self):
    return '< tfu.Function: %s >' % self.name

  def __repr__(self):
    return str(self)


class PyfuncRunner(tf.train.queue_runner.QueueRunner):
  """Load data produced by arbitrary python code.

  Args:
    variables: A dict (to pass to `tfu.make_placeholders()`) describing output of `func()`.
    capacity: Queue capacity, int.
    num_threads: Number of threads, int.
    produces_batches: If true then the queue elements returned by `func()` are entire batches.
    func: Should return either a single training example or a batch of examples (depending on the value of `produces_batches`),
      in the format of a dict with the same keys as `variables` but with the values filled in as numpy arrays.
      Each runner thread will call `func()` independently, so it must be thread-safe.
    args, kwargs: Will be passed to `func()`.

  The runner threads can be paused and the queue can be flushed.

  If `produces_batches`, then the shapes in `variables` need not be fully-defined (as we don't need to call `dequeue_many()`).
  """

  def __init__(self, variables, capacity, num_threads,
               produces_batches, func, *args, **kwargs):
    self.produces_batches = produces_batches
    self.placeholders = make_placeholders(variables)
    self.flat_placeholders = Struct()
    for k, v in self.placeholders.items():
      for i, vv in enumerate(nest.flatten(v)):
        self.flat_placeholders['%s/%d' % (k, i)] = vv

    all_shapes_defined = all(v.get_shape().is_fully_defined()
                             for v in self.flat_placeholders.values())
    if all_shapes_defined:
      # If all shapes are fully-defined, construct the queue_accordingly.
      shapes = [pl.get_shape() for name, pl in self.flat_placeholders.items()]
    else:
      assert produces_batches, 'All shapes must be fully-defined if not queueing batches!'
      shapes = None

    queue = tf.FIFOQueue(capacity, shapes=shapes,
                         names=[name for name,
                                pl in self.flat_placeholders.items()],
                         dtypes=[pl.dtype for name, pl in self.flat_placeholders.items()])
    enqueue_ops = [queue.enqueue(self.flat_placeholders)
                   for _ in range(num_threads)]

    self._num_threads, self._capacity = num_threads, capacity
    self._func, self._args, self._kwargs = func, args, kwargs

    super(PyfuncRunner, self).__init__(queue, enqueue_ops)

    self.queue_size = Function({}, queue.size(), name='queue_size')
    self.dequeue = Function({}, queue.dequeue(), name='dequeue()')

    if produces_batches:
      self.get_batch = Function({}, self.batch(), name='get_batch')

    else:
      batch_size = tf.placeholder(tf.int32, [])
      self.get_batch = Function(dict(batch_size=batch_size),
                                self.batch(batch_size), name='get_batch')

  def _check_cond(self):
    return True

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    return super(PyfuncRunner, self).create_threads(sess, coord, daemon, start)

  @property
  def is_paused(self):
    return not self._flag.is_set()

  def _run(self, sess, enqueue_op, coord=None):
    """Thread main function.

    This is exactly the same as `tf.QueueRunner`, except we enqueue the values generated by `func()`.
    """
    decremented = False
    try:
      do_enqueue = Function(self.placeholders, enqueue_op,
                            session=lambda: sess)

      def enqueue_callable():
        batch = self._func(*self._args, **self._kwargs)
        if batch is not None:
          do_enqueue(**batch)

      prev = 0
      while True:
        if coord and coord.should_stop():
          break

        self._check_cond()
        try:
          enqueue_callable()

        except self._queue_closed_exception_types:
          with self._lock:
            self._runs_per_session[sess] -= 1
            decremented = True
            if self._run_per_session[sess] == 0:
              try:
                sess.run(self._close_op)
              except Exception as e:
                print('Ignored exception: %s' % str(e))

    except Exception as e:
      if coord:
        coord.request_stop(e)
        raise e
      else:
        print('Exception in QueueRunner: %s' % str(e))
        with self._lock:
          self._exceptions_raised.append(e)
        raise
    finally:
      if not decremented:
        with self._lock:
          self._runs_per_session[sess] -= 1

  def batch(self, batch_size=None):
    """Get a batch of tensors."""
    if self.produces_batches:
      assert batch_size is None, 'Cannot enforce a batch size if `func()` returns batches!'
      flat_batch = self._queue.dequeue()
      for name, pl in self.flat_placeholders.items():
        flat_batch[name].set_shape(pl.shape)

    else:
      flat_batch = self._queue.dequeue_many(batch_size)

    batch = Struct()
    for name, pl in self.placeholders.items():
      flat_vals = sorted((k, v)
                         for k, v in flat_batch.items() if k.startswith(name))
      vals = [v for k, v in flat_vals]
      batch[name] = vals[0] if len(
          vals) == 0 else nest.pack_sequence_as(pl, vals)

    return batch


def start_queue_runners(sess):
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord)
  coord._threads = threads
  return coord


def stop_queue_runners(coord):
  coord.request_stop()
  coord.join(coord._threads)
  del coord._threads
