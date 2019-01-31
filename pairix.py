from abc import abstractmethod


class Pairix:
    def __init__(self):
        ''' This module looks for pairwise expressions in the horizon of a predicate.
        '''
        pass

    @abstractmethod
    def annotate(self, *, original_pair=None, predicates_in_horizon=None, graph_coro=None, ):
        '''Annotates the pair, that means expressions that are similar to each other and are distinct from the
           pair, that was found as excluding each other.

        '''
        pass
