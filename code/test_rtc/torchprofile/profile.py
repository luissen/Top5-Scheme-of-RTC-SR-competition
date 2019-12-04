import warnings

from .handlers import handlers
from .utils.trace import trace
import numpy as np
__all__ = ['profile']


def profile(model, args=(), kwargs=None, reduction=sum):
    results = dict()

    graph = trace(model, args, kwargs)
    for node in graph.nodes:
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    results[node] = func(node)
                break


    if reduction is not None:
        xx = np.array(list(results.values()), dtype='float64')
        total_flops = np.sum(xx)
        return total_flops
    else:
        return results
