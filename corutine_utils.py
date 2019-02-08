def coroutine(func):
    ''' Start coroutine with this wrapper, when initializing calling the first time next() on them

    :param func: function with data = (yield)
    :return: started function
    '''
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start
