#------------------------------------------------------------------------------------
# audio_feature_sender -- main
# extract audio features and send features to backend server for training or predict
#
# aixi.wang@hotmail.com
#------------------------------------------------------------------------------------

import pyaudio
#import wave
import sys
import time
import numpy as np
import json
import threading

from edge_methods import Reader
from edge_methods.event_predictor import EventPredictor
from edge_methods.feature_engineer import FeatureEngineer
from edge_methods.majority_voter import MajorityVoter
import Queue

# http://www.dataivy.cn/blog/%E6%B7%B7%E5%90%88%E9%AB%98%E6%96%AF%E6%A8%A1%E5%9E%8Bgaussian-mixture-model_gmm/
import numpy as np
from sklearn import mixture

model = mixture.GMM(n_components=2)
#model.fit(features2)
#model.predict(current_feature)


myqueue = Queue.Queue(maxsize = 1024)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
IDEL_DURATION = RATE
INPUT_INDEX = 1

SMD_ID = 'ACA001'
MSG_ID = '2'

#-------------------
# writefile2
#-------------------
def writefile2(filename,content):
    f = file(filename,'ab')
    fs = f.write(content)
    f.close()
    return 
    
    
#-------------------
# key_word_predict
#-------------------
def key_word_predict(feat):
    print feat
    pass
    
#-------------------
# sender_thread
#-------------------
def sender_thread(p1):
    global myqueue
    print 'sender_thread started!',p1
    
    f = file('fea.csv','ab')
    
    word_status = 0        
    while True:
        feature_list = myqueue.get()

        #print '--------------------------------------'
        #print feature_list
        
        feature_list2 = [feature_list[0][0][0],
                         feature_list[1][0][0],
                         feature_list[2][0][0],
                         feature_list[2][1][0],
                         feature_list[2][2][0],
                         feature_list[2][3][0],
                         feature_list[2][4][0],
                         feature_list[2][5][0],
                         feature_list[2][6][0],
                         feature_list[2][7][0],
                         feature_list[2][8][0],
                         feature_list[2][9][0],
                         feature_list[2][10][0],
                         feature_list[2][11][0],
                         feature_list[2][12][0]]

        #json_out = {'smd_id':SMD_ID,'msg_id':MSG_ID,'data':list(feature_list)}
        #json_out_str = json.dumps(json_out)
        #json_out_str_packed = str(len(json_out_str)) + '\r\n' + json_out_str
        #print 'json_out_str_packed:',json_out_str_packed
        
        s = ''
        for d in feature_list2:
            s += str(d) + ','
            
        s = s.rstrip(',') + '\r\n'
        #print s
        f.write(s)
        

        if feature_list2[1] > 0.01 and word_status == 0:
            print 'start of talk'
            word_status = 1
        
        if feature_list2[1] < 0.01 and word_status == 1:
            print 'end of talk'
            word_status = 0
        
        if word_status == 1:
            key_word_predict(feature_list2)
            
#------------------------------------------
# main
#------------------------------------------
if __name__ == '__main__':
    t1 = threading.Thread(target=sender_thread,args=('test',))
    t1.start()
    
    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    output = False,
                    input_device_index = INPUT_INDEX,                
                    frames_per_buffer = CHUNK)
              
    print "listing..."

    engineer = FeatureEngineer()
    while True:
        try:
        #if 1:
            data = stream.read(CHUNK)
            #print 'data:',data.encode('hex')           
            audio_data_raw = np.fromstring(data,dtype=np.int16)
            audio_data_raw = audio_data_raw/32768.0
            #print type(audio_data_raw)
            #print 'audio_data_raw:',audio_data_raw
            #feature_list1 = signallib.mfcc(audio_data_raw)
            #print 'feature_list1:',feature_list1
            feature_list2 = engineer.feature_engineer_frame(RATE,CHUNK,CHUNK,audio_data_raw)
            #print 'feature_list2:',feature_list2
            myqueue.put(feature_list2)
            
            # Todo -> 
            # 1. put feature to queue
            # 2. send the data to cloud in another thread
            #
            
        except Exception as e:       
            print 'audio exception, retry', str(e)
            time.sleep(1)

