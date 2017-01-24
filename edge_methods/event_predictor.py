# -*- coding: utf-8 -*-

import re
import time

__all__ = [
    'EventPredictor'
]


class EventPredictor:
    """
    Class to classify a new audio signal and determine if it's a baby cry
    """

    def __init__(self, model):
        self.model = model

    def classify(self, new_signal):
        """
        Make prediction with trained model

        :param new_signal: 1d array, 34 features
        :return: feature label string
        """
        t1 = time.time()
        
        category = self.model.predict(new_signal)

        #------------<<<
        print 'predict time cost:',time.time()-t1
        t1 = time.time()
        #------------>>>
        
        # print 'baby_cry_predictor category:',category
        # category is an array of the kind array(['004 - Baby cry'], dtype=object)
        return category


            