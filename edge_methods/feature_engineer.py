# -*- coding: utf-8 -*-

# import pandas as pd
import numpy as np
import time
from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth
# chroma_cens, rmse

__all__ = [
    'FeatureEngineer'
]


class FeatureEngineer:
    """
    Derive features
    """

    RATE = 44100   # All recordings in ESC are 44.1 kHz
    FRAME = 512    # Frame size in samples

    # Features' names
    COL = ['zcr', # 'rms_energy',
           'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
           'mfcc12',
           'sp_centroid', 'sp_rolloff', 'sp_bw'
           # 'chroma1', 'chroma2', 'chroma3', 'chroma4', 'chroma5', 'chroma6', 'chroma7',
           # 'chroma8', 'chroma9', 'chroma10', 'chroma11', 'chroma12'
           ]

    def __init__(self):
        pass

    def feature_engineer(self, audio_data):
        """
        Extract features using librosa.feature.

        Each signal is cut into frames, features are computed for each frame and averaged [median].
        The numpy array is transformed into a data frame with named columns.

        :param audio_data: the input signal samples with frequency 44.1 kHz
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """
        
        t1 = time.time()
        zcr_feat = zero_crossing_rate(y=audio_data, hop_length=self.FRAME)

        #------------<<<
        print 'zero_crossing_rate time cost:',time.time()-t1
        t1 = time.time()
        #------------>>>
        
        # rmse_feat = rmse(y=audio_data, hop_length=self.FRAME)

        mfcc_feat = mfcc(y=audio_data, sr=self.RATE, n_mfcc=13)

        #------------<<<
        print 'mfcc time cost:',time.time()-t1
        t1 = time.time()
        #------------>>>
        
        spectral_centroid_feat = spectral_centroid(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        #------------<<<
        print 'spectral_centrioid time cost:',time.time()-t1
        t1 = time.time()
        #------------>>>
        
        spectral_rolloff_feat = spectral_rolloff(y=audio_data, sr=self.RATE, hop_length=self.FRAME, roll_percent=0.90)

        #------------<<<
        print 'spectral_rolloff time cost:',time.time()-t1
        t1 = time.time()
        #------------>>>

        spectral_bandwidth_feat = spectral_bandwidth(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        #------------<<<
        print 'spectral_bandwidth time cost:',time.time()-t1
        t1 = time.time()
        #------------>>>

        # chroma_cens_feat = chroma_cens(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        concat_feat = np.concatenate((zcr_feat,
                                      # rmse_feat,
                                      mfcc_feat,
                                      spectral_centroid_feat,
                                      spectral_rolloff_feat,
                                      # chroma_cens_feat
                                      spectral_bandwidth_feat
                                      ), axis=0)

        print 'concat_feat:',concat_feat
        return np.mean(concat_feat, axis=1, keepdims=True).transpose()

    def feature_engineer_frame(self,sample_rate,frame_length,hop_length,audio_data): 
        #print 'audio_data len:',len(audio_data)
        #print 'sample_rate:',sample_rate
        #print 'frame_length:',frame_length
        
        #zcr_feat = zero_crossing_rate(y=audio_data, hop_length=frame_length)
        zcr_feat = zero_crossing_rate(y=audio_data,frame_length=frame_length, hop_length=hop_length,center=False)
        # rmse_feat = rmse(y=audio_data, hop_length=self.FRAME)
        #mfcc_feat = mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_feat = mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        
        #spectral_centroid_feat = spectral_centroid(y=audio_data, sr=sample_rate, hop_length=hop_length)       
        #spectral_rolloff_feat = spectral_rolloff(y=audio_data,sr=sample_rate,hop_length=hop_length,roll_percent=0.90)
        #spectral_bandwidth_feat = spectral_bandwidth(y=audio_data,sr=sample_rate,hop_length=hop_length)
        # chroma_cens_feat = chroma_cens(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        feat = [zcr_feat,
              # rmse_feat,
              mfcc_feat,
              #spectral_centroid_feat,
              #spectral_rolloff_feat,
              # chroma_cens_feat
              #spectral_bandwidth_feat
            ]

        return feat
        
        