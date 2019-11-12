# -*- coding: utf-8 -*-
"""
Read & plot history from pickle.
"""
import pickle
import deep_utils

history=pickle.load(open('history.p','rb'))

deep_utils.plot_accuracy(history)
deep_utils.plot_loss(history)
