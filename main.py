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
# sender_thread
#-------------------
def sender_thread(p1):
    global myqueue
    print 'sender_thread started!',p1
    
    f = file('fea.csv','ab')
        
    while True:
        feature_list = myqueue.get()

        #print '--------------------------------------'
        #print type(feature_list)
        
        feature_list2 = [feature_list[0][0][0],
                         feature_list[1][0][0],
                         feature_list[1][1][0],
                         feature_list[1][2][0],
                         feature_list[1][3][0],
                         feature_list[1][4][0],
                         feature_list[1][5][0],
                         feature_list[1][6][0],
                         feature_list[1][7][0],
                         feature_list[1][8][0],
                         feature_list[1][9][0],
                         feature_list[1][10][0],
                         feature_list[1][11][0],
                         feature_list[1][12][0]]

        #json_out = {'smd_id':SMD_ID,'msg_id':MSG_ID,'data':list(feature_list)}
        #json_out_str = json.dumps(json_out)
        #json_out_str_packed = str(len(json_out_str)) + '\r\n' + json_out_str
        #print 'json_out_str_packed:',json_out_str_packed
        
        s = ''
        for d in feature_list2:
            s += str(d) + ','
            
        s = s.rstrip(',') + '\r\n'
        print s
        f.write(s)
  
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
            feature_list = engineer.feature_engineer_frame(RATE,CHUNK,CHUNK,audio_data_raw)
            myqueue.put(feature_list)
            
            # Todo -> 
            # 1. put feature to queue
            # 2. send the data to cloud in another thread
            #
            
        except Exception as e:       
            print 'audio exception, retry', str(e)
            time.sleep(1)

