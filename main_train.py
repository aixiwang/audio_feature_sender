# -*- coding: utf-8 -*-
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

from sklearn import mixture
from sklearn.externals import joblib
from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle


myqueue = Queue.Queue(maxsize = 1024)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
IDEL_DURATION = RATE
INPUT_INDEX = 1

END_POINT_SHRESHOLD = 0.01
WORD_LENGTH = 0.5

SMD_ID = 'ACA001'
MSG_ID = '2'

TRAIN_CYCLE_INIT = 5
TRAIN_CYCLE = 50
TEST_CYCLE = 0

#GMM_N_COMPONETNS = 7

#gmm_model = mixture.GMM(n_components=GMM_N_COMPONETNS)

#-------------------
# train
#-------------------
def train(x_train,y_train):
    
    #pipeline = Pipeline([
    #    ('scl', StandardScaler()),
    #    ('clf', SVC(probability=True))
    #])
        
    #param_grid = [{'clf__kernel': ['linear'], 'clf__C': [1, 1.5, 2, 5]}]
    #estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
    #print 'x_train len:',len(x_train)
    #print 'y_train len:',len(y_train)
    #print 'y_train:',y_train
    #estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
    estimator = SVC()
    svm_model = estimator.fit(x_train, y_train)
    #print 'train features:',features
    #best_estimator = svm_model.best_estimator_
    
    #print 'best_estimator:',best_estimator
    
    # Save model
    #with open(os.path.join(save_path, 'model.pkl'), 'wb') as fp:
    #    pickle.dump(best_estimator, fp)    
    
    joblib.dump(svm_model,'svm.model')
        
#-------------------
# predict
#-------------------
def predict(feature):
    #print 'feature for predict:',feature

    #with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
    #    model = pickle.load(fp)       
    
    model2=joblib.load('svm.model')   
    predict_result = model2.predict(feature)
    #predict_score = model2.score(feature)
    predict_score = [0]
    return predict_result,predict_score   

#-------------------
# writefile2
#-------------------
def writefile2(filename,content):
    f = file(filename,'ab')
    fs = f.write(content)
    f.close()
    return 
    
    
#-------------------
# decode_classify_value
#-------------------
def decode_classify_value(key_word_index):
    if key_word_index == 0:
        print 'shang'
    elif key_word_index == 1:
        print 'xia'
    elif key_word_index == 2:
        print 'zuo'
    elif key_word_index == 3:
        print 'you'
    elif key_word_index == 4:
        print 'qian'
    elif key_word_index == 5:
        print 'hou'
    else:
        print '------'

    #print feat
    pass

#-------------------
# sender_thread
#-------------------
def sender_thread(p1):
    global myqueue
    print 'sender_thread started!',p1
    
    
    f = file('fea.csv','ab')
    f2 = file('fea_word.csv','ab')
    
    word_status = 0

    feature_sets = []
    train_sets = []
    loop_cnt = 0
    loop_cnt_init = 0
    t1 = time.time()
    key_word_index = 0
    train_lable_sets = np.array([])
    lable_sets = []
    total_training_samples = 0
    
    feature_buffer = []
    
    print 'please speak "shang"'
    
    while True:
        feature_list = myqueue.get()

        #print '--------------------------------------'
        #print feature_list
        
        feature_list2 = [feature_list[0][0][0],
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
                         feature_list[2][12][0],
                         feature_list[3][0][0],
                         feature_list[4][0][0],
                         feature_list[5][0][0]]
                         
       
        s = ''
        for d in feature_list2:
            s += str(d) + ','
            
        s = s.rstrip(',') + '\r\n'
        #print s
        f.write(s)
        
        feature_buffer.append(feature_list2)
        if len(feature_buffer) > 3:
            feature_buffer.pop(0)
        #print 'feature_buffer:',feature_buffer    
        
        
        
        
        if feature_list[1][0][0] > END_POINT_SHRESHOLD and word_status == 0:
            #print 'start of talk --------------->'
            if len(feature_buffer) < 3:
                feature_sets.append(feature_list2)
            else:
                #print 'start of talk --------------->'
                feature_sets.append(feature_buffer[1])
                feature_sets.append(feature_buffer[2])

            word_status = 1
            t1 = time.time()
        
        if feature_list[1][0][0] < END_POINT_SHRESHOLD and word_status == 1 and (time.time() - t1) > WORD_LENGTH:        
            #print 'end of talk ---------------<'

            a = np.array(feature_sets)
            meaned_feat = np.mean(a,axis=0)
            #print 'meaned_feat 1:',meaned_feat

            #print 'train_lable_sets:',train_lable_sets
            
            
            if loop_cnt_init < TRAIN_CYCLE_INIT:
                loop_cnt_init += 1
            
            loop_cnt += 1            
            if loop_cnt > TRAIN_CYCLE:
                loop_cnt = 0
            
            if loop_cnt < TRAIN_CYCLE - TEST_CYCLE:
                train_test_status = 1
                print '========> Train loop_cnt:',loop_cnt, ', training_samples:', total_training_samples
                train_sets.append(meaned_feat)
                
                total_training_samples += 1


                
                lable_sets.append(key_word_index)
                
                train_lable_sets = np.array(lable_sets)
                #train_lable_sets = np.reshape(train_lable_sets,(len(train_lable_sets),1))
            
            else:
                print '========> Test loop_cnt:',loop_cnt, ', training_samples:', total_training_samples
                train_test_status = 0
            
            if loop_cnt_init >= TRAIN_CYCLE_INIT:
                if train_test_status == 1:
                    print 'start to train...'
                    #
                    # in train status
                    #
                    #lens = len(train_lable_sets)
                    #train_lable_sets2 = np.reshape(train_lable_sets,(lens,1))

                    
                    train(train_sets,train_lable_sets)
                    #print 'meaned_feat:',meaned_feat
                    
                    meaned_feat2 = np.append(meaned_feat,key_word_index)
                    
                    s = ''
                    for d in meaned_feat2:
                        s += str(d) + ','
                        
                    s = s.rstrip(',') + '\r\n'
                    #print s
                    f2.write(s)
               
                else:
                    print 'start to predict ...'
                    #
                    # in test status
                    #
                    #predict_data_set = [meaned_feat,meaned_feat]
                    meaned_feat = np.reshape(meaned_feat,(1,len(meaned_feat)))
                    result,score = predict(meaned_feat)
                    print 'test result:',result,score
                    decode_classify_value(result)
            
            if train_test_status == 1:
                # key_word list
                key_word_index += 1
                if key_word_index > 5:
                    key_word_index = 0            

                if key_word_index == 0:
                    print 'shang'
                elif key_word_index == 1:
                    print 'xia'
                elif key_word_index == 2:
                    print 'zuo'
                elif key_word_index == 3:
                    print 'you'
                elif key_word_index == 4:
                    print 'qian'
                elif key_word_index == 5:
                    print 'hou'
                else:
                    print '------'
                    
                feature_sets = []
                word_status = 0
        
        #if word_status == 1:
        #    key_word_predict(feature_list2)
            
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

