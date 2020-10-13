# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:27:16 2020

@author: ystseng
"""



import numpy as np
import cv2
import json
from time import time
import matplotlib.pyplot as plt
import os


from RSV_Transformer import RSVTransformer
from matplotlib import cm
from datetime import datetime, timedelta  
import pandas as pd

name_class = ['SpecialPurposeVehicle',
                'Motorcycle',
                'Child',
                'Taxi',
                'Animal',
                'TrafficController',
                'MotorcycleWithRiderWithHelmet',
                'AdultUnderUmbrella',
                'TrailerWithCargo',
                'Adult',
                'ConstructionWorker',
                'FireEngine',
                'PoliceCar',
                'Ambulance',
                'Sedan',
                'Truck',
                'TrailerWithoutCargo',
                'OtherTransport',
                'Cart',
                'ChildUnderUmbrella',
                'Bicycle',
                'Roadblock',
                'Bus',
                'ConstructionVehicle',
                'SmallTruck',
                'Others',
                'BicycleWithRider',
                'MotorcycleWithRiderWithoutHelmet']
dict_class = dict(enumerate(name_class))
inv_dict_class = {v: k for k, v in dict_class.items()}



#%%
## 2. set radar data loader
from RSV_Capture import RadarData

# radar = RadarOffline(filepath=r'C:\\GitHub\\109_RadarFusion\\Dataset\\雷達資料0505\\0505b\\radar.log')
# radar = RadarOffline(filepath=r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0514\0514a\radar.log')
radar = RadarData(filepath=r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cb\radar.log')

radar_data = radar.list_object_frame
radar_data_time_sec = np.array(radar.list_time_total_second)
radar_data_time_period = radar_data_time_sec[1:]-radar_data_time_sec[:-1]
period_dict = {val:ii for ii,val in enumerate(set(radar_data_time_period))}
inv_period_dict = {v: k for k, v in period_dict.items()}
counts = np.bincount([period_dict[ii] for ii in radar_data_time_period ])
radar_fps = inv_period_dict[np.argmax(counts)]

#%%
rt = RSVTransformer()
rt.load_parametre(path_load='RSV_metrix.npz')
#%%
from random import random 
class Trajectory():
    def __init__(self):
        self.transform = None
        self.trajectory = {}
        self.speed = {}
        self.bbox = {}
        self.length_histogram = {}
        self.object_types = {}
        self.color = tuple([int(x *255) for x in  cm.get_cmap('hsv')(int(random()*255))[0:3]])
    
    def get_path(self):
        '''取得雷達的所有路徑'''
        return np.array(list(self.trajectory.values()))
    
    def get_time(self):
        '''取得當下對應的時間點'''
        return np.array(list(self.trajectory.keys()))
    
    @staticmethod
    def smooth_trajectory(path, smooth_window=5):
        '''用來平滑軌跡路徑'''        
        if smooth_window%2==0:
            smooth_window=smooth_window+1
        end_nodes= (smooth_window-1)/2 if smooth_window %2 else smooth_window/2-1
        
        path_x = path[:,0]
        path_x = np.concatenate([path_x[0]*np.ones(int(end_nodes)), path_x, path_x[-1]*np.ones(int(end_nodes)), ],axis=0)
        moving_average_path_x = np.convolve(path_x  , np.ones(smooth_window), 'valid')/smooth_window
        
        path_y = path[:,1]
        path_y = np.concatenate([path_y[0]*np.ones(int(end_nodes)), path_y, path_y[-1]*np.ones(int(end_nodes)), ],axis=0)
        moving_average_path_y = np.convolve(path_y  , np.ones(smooth_window), 'valid')/smooth_window
        
        moving_average_path  = np.stack([moving_average_path_x, moving_average_path_y],axis=1)        
        
        return moving_average_path
    
    def get_smooth_path(self, smooth_window=5):
        '''用來取得平滑過的雷達資料'''
        path_node = self.get_path()
        path_smooth_node = self.smooth_trajectory(path_node, smooth_window)
        return path_smooth_node 
    
    def get_node_at(self, radar_timestamp):
        '''取得某時間點的雷達位置'''
        get_timestamp = datetime.strptime(radar_timestamp, "%Y-%m-%d %H:%M:%S.%f").timestamp()*1000
        index_time = np.argmin(np.abs(self.get_time()- get_timestamp))
        return self.get_path()[index_time,:]
        
    def get_velocity(self, time_line=-1):
        #TODO 計算軌跡速度用
        '''
        if time_line ==-1:
            radar_path = self.get_smooth_path(10)
            radar2image_path = rt.transepose_radar_to_image(radar_path)
            radar_path[1:]-radar_path[:-1]
        '''

class RadarTrajectory(Trajectory):
    def __init__(self, oid=None):
        Trajectory.__init__(self)
        self.oid = oid
        self.space = 'radar'
        self.last_update = None
        self.fusion_obj = {}
    
    def get_fusion_id(self,):
        if len(self.fusion_obj)==0:
            return self.oid
        else:
            return np.argmax(np.bincount(list(self.fusion_obj.values())))
        
    def update_new_state(self, radar_frame_object, current_timestamp):
        '''從雷達以及當下frame時間更新雷達物件的位置'''
        self.last_update = current_timestamp
        self.trajectory.update({ current_timestamp: (radar_frame_object['x'], radar_frame_object['y']) })
        self.speed.update({ current_timestamp: (radar_frame_object['speed_x'], radar_frame_object['speed_y']) })
        self.bbox.update({ current_timestamp: [[None,None], [None,None], [None,None], [None,None]] })
        if radar_frame_object['length'] not in self.length_histogram:
            self.length_histogram.update({radar_frame_object['length']:0})
        self.length_histogram[radar_frame_object['length']] = self.length_histogram[radar_frame_object['length']]+1
        if radar_frame_object['type'] not in self.object_types:
            self.object_types.update({radar_frame_object['type']:0})
        self.object_types[radar_frame_object['type']] = self.object_types[radar_frame_object['type']]+1
    
    def set_transformer_object(self,transform):
        self.transform = transform
        
    def get_path_I(self,):
        radar_path = self.get_path()
        if self.transform:
            radar_path_on_image = self.transform.transepose_radar_to_image(radar_path)
            return radar_path_on_image
        else:
            print('Dont get transformer object, return raw radar path')
            return radar_path
        
    def get_smooth_path_I(self,smooth_window):
        radar_smooth_path = self.get_smooth_path(smooth_window)
        if self.transform:
            radar_smooth_path_on_image = self.transform.transepose_radar_to_image(radar_smooth_path)
            return radar_smooth_path_on_image 
        else:
            print('Dont get transformer object, return raw smooth radar path')
            return radar_smooth_path 
            
    '''
radar_object = radar_tracker.live_radar_track[167].get_path()
radar2image_path = rt.transepose_radar_to_image(radar_object)
detection_object = radar_tracker.live_detection_track[27].get_path()

plt.figure()
plt.plot(radar2image_path[:,0], radar2image_path[:,1],'o',label='radar');plt.axis([0,1920,1080,0])
plt.plot(detection_object[:,0], detection_object[:,1],'o',label='detection');plt.axis([0,1920,1080,0])
plt.legend()
radar = radar_tracker.live_radar_track[167]
detect = radar_tracker.live_detection_track[27]
intersation_time = np.intersect1d(radar.get_time(), detect.get_time())
value_radar_path =np.array([radar_tracker.live_radar_track[167].trajectory[x] for x in intersation_time ])
value_radar_path  = rt.transepose_radar_to_image(value_radar_path )
value_detection_path =np.array([radar_tracker.live_detection_track[27].trajectory[x] for x in intersation_time ])
plt.figure()
plt.plot(value_radar_path [:,0], value_radar_path [:,1],'o',label='radar');plt.axis([0,1920,1080,0])
plt.plot(value_detection_path [:,0], value_detection_path [:,1],'o',label='detection');plt.axis([0,1920,1080,0])
plt.legend()
'''

class DetectionTrajectory(Trajectory):
    def __init__(self, oid=None, transfromer=None):
        Trajectory.__init__(self)
        self.oid = oid
        self.transfromer = transfromer
        self.space = 'video'
        self.last_update = None
        self.fusion_obj = {}
            
    def get_fusion_id(self, current_time):
        ## current_time = read_radar_time
        # self = radar_tracker.live_detection_track[]
        # current_timestamp = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f").timestamp()*1000
        if len(self.fusion_obj)==0:
            return np.nan
        else:
            
            HO_Class = np.bincount(list(self.fusion_obj.values())[-10:])
            HO_Class = HO_Class *(HO_Class >3)
            if sum(HO_Class)>0:
                return np.argmax(HO_Class)
            else:
                return np.nan
            # return np.argmax(np.bincount(list(self.fusion_obj.values())))
        
    def update_new_state(self, detection_frame_object, current_timestamp):
        # detection_frame_object = currentTracker[0]# Debug
        self.last_update = current_timestamp
        bbox = np.array(detection_frame_object[4][-1])
        self.bbox.update({ current_timestamp: [[bbox[0],bbox[1]], [bbox[2],bbox[1]], [bbox[2],bbox[3]], [bbox[0],bbox[3]]] })
        self.trajectory.update({ current_timestamp: ((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2) })
        # self.speed.update({ current_timestamp: (detection_frame_object['speed_x'], detection_frame_object['speed_y']) })
        # if radar_frame_object['length'] not in self.length_histogram:
        #     self.length_histogram.update({radar_frame_object['length']:0})
        # self.length_histogram[radar_frame_object['length']] = self.length_histogram[radar_frame_object['length']]+1
        if detection_frame_object[3] not in self.object_types:
            self.object_types.update({detection_frame_object[3]:0})
        self.object_types[detection_frame_object[3]] = self.object_types[detection_frame_object[3]]+1
        

        
class RadarTracker():
    def __init__(self,):
        self.live_radar_track = {}
        self.dead_radar_track = {}
        self.live_detection_track = {}
        self.dead_detection_track = {}
        self.radar_track_patients = {}
        self.detection_track_patients = {}
        self.number_patient = 4
        self.lastupdate = None
        self.timecost = {'find':[], 'loss':[], 'radar':[], 'detec':[]}
        self.loss_time ={}
        
    def set_RSV_tranfsformer(self, transfromer):
        self.transfromer = transfromer
        
    def update_radar_tracker(self, radar_objects, read_radar_time):
        '''更新當下frame的所有雷達
        radar_objects: 雷達當下frame的所有時間點
        radar_objects, read_radar_time = radar_points, read_radar_time
        '''
        radar_objects = radar_points.copy() #Debug
        ## Get current time in milisecond
        current_timestamp = datetime.strptime(read_radar_time, "%Y-%m-%d %H:%M:%S.%f").timestamp()*1000
        self.lastupdate = current_timestamp
        del(radar_objects['time'])
        
        ## Update radar tracker with cruuent time
        self.current_radar_ids = []
        for radar_i in radar_objects.keys():
            # get ID
            radar_objevt_id = radar_objects[radar_i]['oid']
            # if not exist, make a new tracker
            if radar_objevt_id not in self.live_radar_track:
                self.live_radar_track.update({ radar_objevt_id: RadarTrajectory(radar_objevt_id) })
                
                self.radar_track_patients.update({ radar_objevt_id: 0 })
            # update tracker
            radar_frame_object = radar_objects[radar_i]
            self.live_radar_track[radar_objevt_id].update_new_state(radar_frame_object, current_timestamp)
            self.current_radar_ids.append(radar_objevt_id)
        
        # Accumulate disappear tracker
        for live_id in self.live_radar_track.keys():
            if live_id not in self.current_radar_ids:
                self.radar_track_patients[live_id] += 1
                
    def update_detection_tracker(self, detection_tracker, read_radar_time):
        '''更新當下frame的所有影像測結果
        detection_tracker: 當下frame的所有時間點
        '''
        detection_objects = Tracker.alive_tracker.copy() #Debug
        ## Get current time in milisecond
        current_timestamp = datetime.strptime(read_radar_time, "%Y-%m-%d %H:%M:%S.%f").timestamp()*1000
        self.lastupdate = current_timestamp
        ## Update radar tracker with cruuent time
        self.current_detection_ids = []
        for detection_object in detection_objects:
            # detection_object = detection_objects[0]
            # get ID
            detection_objevt_id = detection_object[0]
            # if not exist, make a new tracker
            if detection_objevt_id not in self.live_detection_track:
                self.live_detection_track.update({ detection_objevt_id : DetectionTrajectory(detection_objevt_id) })
                self.detection_track_patients.update({ detection_objevt_id : 0 })
            # update tracker
            self.live_detection_track[detection_objevt_id].update_new_state(detection_object, current_timestamp)
            self.current_detection_ids.append(detection_objevt_id)
        
        # Accumulate disappear tracker
        for live_id in self.live_detection_track.keys():
            if live_id not in self.current_detection_ids:
                self.detection_track_patients[live_id] += 1

    def update_dead_tracker(self):
        '''用來清理確定死掉的tracker'''
        # Decide dead radar tracker with  =patient threadhold
        drop_ids = []
        for live_id in self.radar_track_patients.keys():
            if self.radar_track_patients[live_id] >= self.number_patient:
                drop_ids.append(live_id)
        # Update dead tracker 
        for drop_id in drop_ids:
            self.dead_radar_track.update({ drop_id: self.live_radar_track[drop_id] })
            del(self.radar_track_patients[drop_id])
            del(self.live_radar_track[drop_id])
            
        # Decide dead radar tracker with patient threadhold
        drop_ids = []
        for live_id in self.live_detection_track.keys():
            if self.detection_track_patients[live_id] >= self.number_patient:
                drop_ids.append(live_id)
                
        # Update dead tracker 
        for drop_id in drop_ids:
            self.dead_detection_track.update({ live_id: self.live_detection_track[drop_id] })
            del(self.detection_track_patients[drop_id])
            del(self.live_detection_track[drop_id])
            
            
    # def calculate_overlap_time(self, source_tracker:object, target_tracker:object):
    #     '''以track物件取得兩軌跡的重疊的時間'''
    #     source_time = source_tracker.get_time()
    #     target_time = target_tracker.get_time()
    #     intersect_time = np.intersect1d(source_time , target_time)
    #     source_tracker_overlap = np.array([source_tracker.trajectory[x] for x in intersect_time])
    #     target_tracker_overlap = np.array([target_tracker.trajectory[x] for x in intersect_time])
    #     return source_tracker_overlap, target_tracker_overlap, intersect_time
    def calculate_overlap_time(self, source_tracker:object, target_tracker:object):
        '''以track物件取得兩軌跡的重疊的時間'''

        source_trajectory = dict(zip(source_tracker.get_time(), source_tracker.get_smooth_path(5)))
        target_trajectory = dict(zip(target_tracker.get_time(), target_tracker.get_smooth_path(5)))
        intersect_time = np.intersect1d(source_tracker.get_time() , target_tracker.get_time())

        source_tracker_overlap = np.array([source_trajectory [x] for x in intersect_time])
        target_tracker_overlap = np.array([target_trajectory [x] for x in intersect_time])
        return source_tracker_overlap, target_tracker_overlap, intersect_time
    
    def calculate_velocity(self, trajectory:np.array):
        '''以軌跡點取得方向'''
        # trajectory  = target_trajectory_overlap
        
        A = np.vstack([trajectory[:,0], np.ones(len(trajectory[:,0]))]).T
        y = trajectory[:,1]
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        direction = np.arctan(m)*180/np.pi + [180 if trajectory[-1][0] - trajectory[0][0]<0 else 0][0]
        return direction 
    
    def calculate_static(self, trajectory:np.array, threadhold_range=2):
        '''以軌跡點判斷是否為靜止狀態'''
        
        # trajectory = self.live_detection_track[3].get_path()
        # trajectory1 = self.live_detection_track[44].get_path()
        # plt.figure();plt.plot(trajectory[:,0], trajectory[:,1]);plt.plot(trajectory1[:,0], trajectory1[:,1])
        # np.mean(trajectory[1:]-trajectory[:-1], axis= 0)
        # np.mean(trajectory1[1:]-trajectory1[:-1], axis= 0)
        
        # (trajectory[-1]-trajectory[0])/len(trajectory)
        # (trajectory1[-1]-trajectory1[0])/len(trajectory1)

        # np.std(trajectory, axis =0)
        # np.std(trajectory1, axis =0)
        
        
        is_static = np.linalg.norm(np.mean(trajectory[1:]-trajectory[:-1], axis= 0)) < threadhold_range
        return is_static
    
    def get_path_loss(self, source_tracker:object, target_tracker:object):
        '''以追蹤物件判斷兩物件的相似度'''
        # source_tracker = radar_tracker.live_radar_track[200]
        # target_tracker = radar_tracker.live_detection_track[280]
        t1 = time()
        self.observer_frame = 30
        ## get intersection time 
        source_trajectory_overlap, target_trajectory_overlap, intersect_time = self.calculate_overlap_time(source_tracker, target_tracker)

        source_trajectory_overlap = source_trajectory_overlap[-self.observer_frame:, :]
        target_trajectory_overlap = target_trajectory_overlap[-self.observer_frame:, :]

        t2 = time()
        ## if tracker is radar type, do the project to image
        if source_tracker.space == 'radar':
            source_trajectory_overlap = self.transfromer.transepose_radar_to_image(source_trajectory_overlap)
        if target_tracker.space == 'radar':
            target_trajectory_overlap = self.transfromer.transepose_radar_to_image(target_trajectory_overlap)
        t3 = time()  
        
        if (len(source_trajectory_overlap)<=1) or (len(target_trajectory_overlap )<=1):
            return np.nan,np.nan, np.nan
        
        ## calculate loss
        t4 = time()
        loss_distance = np.sum(np.abs(source_trajectory_overlap - target_trajectory_overlap ))/len(source_trajectory_overlap)
        t5 = time()
        loss_direction = self.calculate_velocity(target_trajectory_overlap) - self.calculate_velocity(source_trajectory_overlap)
        t6 = time()
        self.loss_time.update({ 'get_path':t2-t1, 'project_path':t3-t2, 'decide_length':t4-t3, 'loss_distance ':t5-t4, 'loss_direction ':t6-t5})
        
        # plt.figure();plt.plot(source_trajectory_overlap[:,0], source_trajectory_overlap[:,1]);plt.plot(target_trajectory_overlap[:,0], target_trajectory_overlap[:,1])
        return loss_distance, loss_direction
    
    def find_similarity_pairs(self, current_time):
        '''已給定的時間，判斷目前物件的對應情況，輸出雷達與影像的對應ID'''
        ## current_time = read_radar_time
        current_timestamp = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f").timestamp()*1000
        # result = pd.DataFrame(columns=['detect', 'radar', 'distance_loss', 'direction_loss'])        
        DETECT = 0 
        RADAR = 1
        LOSS_DISTANCE = 2
        LOSS_DIRECTION = 3

        PAIR = 5 
        TOTAL_LOSS =6
        TH_DISTANCE = 150
        TH_DIRECT = 12
        t1 = time()
        
        ## get 10 second 
        recent_radars = {x:self.live_radar_track[x] for x in self.live_radar_track 
                         if  ((current_timestamp-self.live_radar_track[x].last_update)<10*1000)&
                              (self.live_radar_track[x].get_path()[-1][0]<70)
                         }

        # recent_detects = {x:self.live_detection_track[x] for x in self.live_detection_track
        #                   if  ((current_timestamp-self.live_detection_track[x].last_update)<10*1000) &
        #                        ~self.calculate_static(self.live_detection_track[x].get_path(), 4)
        #                   }
        appear_detects = {x:self.live_detection_track[x] for x in self.live_detection_track 
                          if  ((current_timestamp==self.live_detection_track[x].last_update)<500) &
                              ~self.calculate_static(self.live_detection_track[x].get_path(), 2)
                          }
        t2 = time()

        ## 計算個物件的相似度，並去除一部分太遠的物件
        result_raw = []
        for recent_radar_id in recent_radars :
            recent_radar = self.live_radar_track[recent_radar_id]
            # for recent_detect_id in recent_detects:
            for recent_detect_id in appear_detects:
                recent_detect = self.live_detection_track[recent_detect_id]
                if (len(recent_detect.trajectory)>5) & (len(recent_radar.trajectory)>5):
                    # print(recent_detect_id)
                    distance_loss, direction_loss = self.get_path_loss(recent_radar, recent_detect)
                    is_appear = 1 if recent_detect_id in list(appear_detects.keys()) else 0
                    is_out_range = 999 #if (distance_loss>=TH_DISTANCE*2)|(direction_loss>=TH_DIRECT) else np.nan
                    
                    result_raw.append([recent_detect_id, recent_radar_id, distance_loss, direction_loss, is_appear, is_out_range, np.nan])
                    # result.loc[i,['detect', 'radar', 'distance_loss', 'direction_loss']]= [recent_detect_id, recent_radar_id, distance_loss, direction_loss ]
                    # i+=1
        t3 = time()
        # print(t3-t2)
        
        ## 判斷雷達與物件框對應的物件
        if len(result_raw)==0:
            return {}
        else:
            
            result = np.array(result_raw)
            ## 讓每個雷達，找到軌跡最接近的物件框。並且相似度誤差不能低於200%
            t3 = time()
            for radar_id in np.unique(result[:,RADAR]):
                # radar_id  = 168
                remain_raw = np.where(  (result[:,RADAR]==radar_id) &(result[:,LOSS_DISTANCE] < 2*TH_DISTANCE ) & (np.abs(result[:,LOSS_DIRECTION]) < 2*TH_DIRECT))[0]
                if len(remain_raw )>0:
                    remain_target = result[remain_raw,:]
                    loss = (remain_target[:,LOSS_DISTANCE]/TH_DISTANCE*100 + np.abs(remain_target[:,LOSS_DIRECTION])/TH_DIRECT*50)
                    
                    min_loss_row = np.argmin(loss)
                    most_close_row = remain_raw[np.argmin(loss)]
                    if (result[most_close_row,TOTAL_LOSS] < loss[min_loss_row]) | (loss[min_loss_row]>200):
                        result[remain_raw ,PAIR] = 999
                        continue
                    ## 如果搶到別人的`，先讓人回去找第二順位
                    if sum(result[result[:,DETECT]==remain_target[min_loss_row, DETECT],PAIR]!=999):
                        refind_radar = result[result[:,DETECT]==remain_target[min_loss_row, DETECT],PAIR]
                        refind_radar = refind_radar[refind_radar!=999][0]
                        remain_raw_sec = np.where(  (result[:,RADAR]==refind_radar) & (result[:,PAIR]==999) &(result[:,LOSS_DISTANCE] < 2*TH_DISTANCE ) & (np.abs(result[:,LOSS_DIRECTION]) < 2*TH_DIRECT))[0]
                        remain_target_sec = result[remain_raw_sec,:]
                        loss_sec = (remain_target_sec[:,LOSS_DISTANCE]/TH_DISTANCE*100 + np.abs(remain_target_sec[:,LOSS_DIRECTION])/TH_DIRECT*50)
                        if len(loss_sec):
                            min_loss_row_sec = np.argmin(loss_sec )
                            most_close_row_sec = remain_raw_sec[np.argmin(min_loss_row_sec)]
                            if ~(result[most_close_row_sec,TOTAL_LOSS] < loss_sec[min_loss_row_sec]) | (loss_sec[min_loss_row_sec]>200):
                                # result[remain_raw ,PAIR] = 999
                                result[most_close_row_sec ,TOTAL_LOSS] = loss_sec[min_loss_row_sec]
                                result[most_close_row_sec ,PAIR] = refind_radar 
                    ## 更新屬於自己的
                    result[result[:,DETECT]==remain_target[min_loss_row,DETECT],TOTAL_LOSS] = loss[min_loss_row]
                    result[(result[:,DETECT]==remain_target[min_loss_row,DETECT]), PAIR] = 999
                    result[most_close_row, PAIR] = radar_id
                    not_close_id = list(remain_raw)
                    # not_close_id.remove(most_close_row)
                    # result[not_close_id,PAIR] = 999
            t4 = time()
        
            for appear_detection_id in appear_detects.keys():
                # appear_detection_id = 280
                appear_row = np.where( (result[:,PAIR]==999) & 
                                      (result[:,DETECT]==appear_detection_id) & 
                                      (result[:,LOSS_DISTANCE]<TH_DISTANCE ) &
                                      (np.abs(result[:,LOSS_DIRECTION])<TH_DIRECT) )[0]
                if len(appear_row )>0:   
                    appear_target = result[appear_row,:]
                    loss = np.abs(appear_target[:,LOSS_DISTANCE]/TH_DISTANCE*100 + np.abs(appear_target[:,LOSS_DIRECTION])/TH_DIRECT*100)
                    min_loss_row = np.argmin(loss)
                    most_close_row = appear_row[min_loss_row]
                    result[most_close_row,PAIR] = appear_target[min_loss_row ,RADAR]
                    not_close_id = list(appear_row)
                    not_close_id.remove(most_close_row)
                    result[not_close_id,PAIR] = 999
                
            t5 = time()    
            dict_fusion_pair = dict(zip(result[result[:,PAIR]!=999,RADAR],result[result[:,PAIR]!=999,DETECT]))
            
            for radar_id in dict_fusion_pair:
                self.live_radar_track[radar_id].fusion_obj.update({current_timestamp:dict_fusion_pair[radar_id ]})
                self.live_detection_track[dict_fusion_pair[radar_id]].fusion_obj.update({current_timestamp:radar_id })
            print('find:%f, loss:%f, radar:%f, detec:%f '%((t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, (t5-t4)*1000))
            self.timecost['find'].append(t2-t1)
            self.timecost['loss'].append(t3-t2)
            self.timecost['radar'].append(t4-t3)
            self.timecost['detec'].append(t5-t4)

            return dict_fusion_pair 

# self.live_detection_track[34].get_fusion_id()
# self.live_radar_track[167].get_fusion_id()
# np.mean(self.timecost['find'])
# np.mean(self.timecost['loss'])
# np.mean(self.timecost['radar'])
# np.mean(self.timecost['detec'])

# plt.figure();
# plt.imshow(frame_show[:,:,::-1])
# plt.figure();plt.axis([0,1020,1920,0])
# plt.plot(source_trajectory_overlap [:,0], source_trajectory_overlap [:,1])
# plt.plot(target_trajectory_overlap_I [:,0], target_trajectory_overlap_I [:,1])    
#%%        
# from time import time
## . read viedo and Project radar data and detection
targetid = [168,212,223,1,14,39,56,70,78,85,89]

# A.read detection result 
with open(r'C:\GitHub\109_RadarFusion\detection_0604b.txt') as f:
    data = f.readlines()
detections = json.loads(data[0])

# B.read video stream
cap = cv2.VideoCapture(r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cb\2020-06-04_16-46-28,448645.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
video_start_time = datetime(2020, 6, 4, 16, 46, 28, 70)
video_delay_time = timedelta(seconds=0, microseconds = 0 )

# C.set save video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
path_result = 'C:\\GitHub\\109_RadarFusion'
out_d = cv2.VideoWriter(os.path.join(path_result,'radar_detection_fusion_0604.mp4'), fourcc, 30, (1920,1080))

# D.set radar tracker 
radar_tracker = RadarTracker()
radar_tracker.set_RSV_tranfsformer(rt)
self =radar_tracker

# E.Tracker
from test_tracking import object_Tracker
Tracker = object_Tracker(1920, 1080, dict_class) #init tracker

# F.result 
dataframe_objects = pd.DataFrame(columns=['time','ID', 'type', 'class', 'x1', 'y1','x2', 'y2', 'cx', 'cy'])

time_total = []
for num_frame in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
# for num_frame in range(170, 224):

    currenttime = video_start_time + timedelta(microseconds  = 1/fps*10**6 )*num_frame
    
    ## A.read video image
    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame )
    ret,frame = cap.read()
    frame_show = frame.copy()
    
    ## B.read radar data 
    read_radar_time = (currenttime-video_delay_time).strftime("%Y-%m-%d %H:%M:%S.%f") 
    radar_points= radar.read( read_radar_time )  
    
    ## C.read detection 
    detection = detections['%s'%num_frame] 
    det_result=[]
    for i_ID in detection.keys():
        current_object = detection[i_ID]
        bbox = [int(ii) for ii in current_object[1][1]]
        x, y, w, h = bbox[0],\
            bbox[1],\
            bbox[2],\
            bbox[3]
        xmin = int(round(x - (w / 2))*1920/608)
        xmax = int(round(x + (w / 2))*1920/608)
        ymin = int(round(y - (h / 2))*1080/608)
        ymax = int(round(y + (h / 2))*1080/608)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        det_result.append([inv_dict_class[current_object[0]], current_object[1][0],[xmin, ymin,xmax, ymax]])
        
    ## D.tracking
    if len(Tracker.alive_tracker)>0:
        # print ("Update tracker no dlib")
        Tracker.update_tracker_list_no_dlib(det_result)
    else:
        # print ("Start tracker no dlib")   
        Tracker.start_tracking_no_dlib(det_result)
        
    ## E.update trackers 
    # radar 
    radar_tracker.update_radar_tracker(radar_points, read_radar_time )
    radar_tracker.update_dead_tracker()
    
    #detection
    radar_tracker.update_detection_tracker(Tracker.alive_tracker, read_radar_time )
    
    ## F.calculate fusion IDs
    t1 = time()
    dict_RD_pair = radar_tracker.find_similarity_pairs(read_radar_time)
    dict_DR_pair = {dict_RD_pair[x]:x for x in dict_RD_pair}
    # print(time()-t1)
    time_total.append(time()-t1)
    #### Show Result
    ## A.radar 
    for radar_object_id in radar_tracker.live_radar_track.keys():
        radar_object = radar_tracker.live_radar_track[radar_object_id]
        image_path = rt.transepose_radar_to_image(radar_object.get_smooth_path(20))
        # image_path = rt.transepose_radar_to_imageradar_object.get_path())
        markersize = 5
        if  not (0<image_path[-1][0]<1920 and 0<image_path[-1][1]<1080):
            continue
        for pp in image_path[::-1][:50]:
            cv2.circle(frame_show , (int(pp[0]), int(pp[1])), int(markersize ), radar_tracker.live_radar_track[radar_object_id].color,-1)
            markersize *=0.95
        cv2.putText(frame_show , '%s'%radar_object_id , (int(image_path[-1][0]), int(image_path[-1][1])), cv2.FONT_HERSHEY_PLAIN, 
                    1, (0,0,255),1,cv2.LINE_4)
        
        radar_templat = pd.DataFrame(columns=['time','ID','type','cx', 'cy', 'x1', 'y1','x2', 'y2', 'length'])
        radar_templat.loc[currenttime,['time','ID','type','cx', 'cy']]=[currenttime, radar_object.oid, 'radar', image_path[-1][0], image_path[-1][1]]
        dataframe_objects = dataframe_objects.append(radar_templat, ignore_index=True)
        
    cv2.imshow('ret', frame_show )
    
    ## C.show radar on image 
    # for r_ID in [i for i in radar_points.keys() if i != 'time'] :
    #     if radar_points[r_ID]['oid'] not in []:
    #         radar_point = np.array([[radar_points[r_ID]['x'], radar_points[r_ID]['y']]], dtype =np.float32)
    #         radar2image_points = rt.transepose_radar_to_image(radar_point )[0] ## x,y
    #         if  not (0<radar2image_points[1]<1920 and 0<radar2image_points[0]<1080):
    #             continue
    #         if radar_points[r_ID]['oid'] in targetid:
    #             cv2.circle(frame_show , (int(radar2image_points[0]), int(radar2image_points[1])), int(radar_points[r_ID]['length']*3), (0,0,255),-1)
    #         else:
    #             cv2.circle(frame_show , (int(radar2image_points[0]), int(radar2image_points[1])), int(radar_points[r_ID]['length']*1), (0,255,255),-1)
    #         cv2.putText(frame_show , '%s'%(radar_points[r_ID]['oid']), (int(radar2image_points[0]), int(radar2image_points[1])), cv2.FONT_HERSHEY_PLAIN, 
    #                 1, (0,0,255),1,cv2.LINE_4)
    frame_show_resize = frame_show.copy()
    
    
    ## E.show detection on image
    # det_result=[]
    # for i_ID in detection.keys():
    #     current_object = detection[i_ID]
    #     bbox = [int(ii) for ii in current_object[1][1]]
    #     x, y, w, h = bbox[0],\
    #         bbox[1],\
    #         bbox[2],\
    #         bbox[3]
    #     xmin = int(round(x - (w / 2))*1920/608)
    #     xmax = int(round(x + (w / 2))*1920/608)
    #     ymin = int(round(y - (h / 2))*1080/608)
    #     ymax = int(round(y + (h / 2))*1080/608)
    #     pt1 = (xmin, ymin)
    #     pt2 = (xmax, ymax)
    #     cv2.rectangle(frame_show, pt1, pt2, (0, 150, 0), 2)
    #     det_result.append([inv_dict_class[current_object[0]], current_object[1][0],[xmin, ymin,xmax, ymax]])
        
        


    for i_detobj, detect_object in enumerate(Tracker.alive_tracker):
        cv2.rectangle(frame_show, tuple(detect_object[4][-1][0:2]), tuple(detect_object[4][-1][2:4]), (0, 185, 0), 2)    
        current_bbox = detect_object[4][-1]

        funsion_object = radar_tracker.live_detection_track[detect_object[0]]
        fusion_radar  = funsion_object.get_fusion_id(read_radar_time)
        if ~np.isnan(fusion_radar):
            show_color = [radar_tracker.live_radar_track[fusion_radar].color if (fusion_radar in radar_tracker.live_radar_track) else radar_tracker.dead_radar_track[fusion_radar].color][0]
            cv2.rectangle(frame_show, tuple(detect_object[4][-1][0:2]), tuple(detect_object[4][-1][2:4]), show_color , 4)
        
        # if detect_object[0] in list(dict_DR_pair):
        #     show_color = radar_tracker.live_radar_track[dict_DR_pair[detect_object[0]]].color  
        #     cv2.rectangle(frame_show, tuple(detect_object[4][-1][0:2]), tuple(detect_object[4][-1][2:4]), show_color , 4)
        cv2.putText(frame_show , '%s'%detect_object[0], tuple(detect_object[4][-1][0:2]), cv2.FONT_HERSHEY_PLAIN, 
                    1, (0,0,255),2,cv2.LINE_4)
        
        detection_templat = pd.DataFrame(columns=['time', 'ID','type','cx', 'cy','x1', 'y1','x2', 'y2', 'class'])
        detection_templat.loc[currenttime,['time', 'ID','type','cx', 'cy', 'class']] = [currenttime,detect_object[0], 'detection', (current_bbox[0]+current_bbox[2])/2, (current_bbox[1]+current_bbox[3])/2, detect_object[3]]
        detection_templat.loc[currenttime,['x1', 'y1','x2', 'y2']] = current_bbox 
        dataframe_objects = dataframe_objects.append(detection_templat, ignore_index=True)
    cv2.putText(frame_show , '%s'%currenttime, (20,120), cv2.FONT_HERSHEY_PLAIN, 
                    2, (0,0,255),2,cv2.LINE_4)      
    cv2.putText(frame_show , '%s'%num_frame, (20,75), cv2.FONT_HERSHEY_PLAIN, 
                        2, (0,255,255),2,cv2.LINE_4)                
    
    ## F.show resize 
    cv2.putText(frame_show_resize , '%s'%currenttime, (20,120), cv2.FONT_HERSHEY_PLAIN, 
                    2, (0,0,255),2,cv2.LINE_4)      
    cv2.putText(frame_show_resize , '%s'%num_frame, (20,50), cv2.FONT_HERSHEY_PLAIN, 
                        1, (0,0,255),1,cv2.LINE_4)                
    
    frame_show_resize= cv2.resize(frame_show_resize, (608,608))
    for i_ID in detection.keys():
        current_object = detection[i_ID]
        bbox = [int(ii) for ii in current_object[1][1]]
        x, y, w, h = bbox[0],\
            bbox[1],\
            bbox[2],\
            bbox[3]
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(frame_show_resize, pt1, pt2, (0,255,0), 2)
        

        
    cv2.rectangle(frame_show_resize, (500,50),(532,50+32),(155, 0, 155),-1)
    cv2.imshow('ret_resize', frame_show_resize)    
    cv2.imshow('ret', frame_show)
    out_d.write(frame_show)
    k = cv2.waitKey(1)
    if k == 27:         # wait for ESC key to exit17
        # while(1):
        #     k = cv2.waitKey(1)
        #     if k ==13:
                break
    
out_d.release()
dataframe_objects.to_csv('fusion.csv')


'''


一開始的小黃
(160)
0
5
6

停車場機車
(161)

一開始左轉機車
(162)
7

第二台藍色汽車
(163)
8
16

汽車
(164)
(170)轉彎IDs
15

遠方的腳踏車
(165)


警車
(166)
23
#%%
貨車
(167)
27

trunk_det = radar_tracker.live_detection_track[27].get_path()
trunk_det_time = radar_tracker.live_detection_track[27].get_time()
trunk_rad = radar_tracker.live_radar_track[167].get_path_I()
trunk_rad2i = rt.transepose_radar_to_image(trunk_rad)
trunk_rad2i_time = radar_tracker.live_radar_track[167].get_time()
plt.figure();plt.axis([0,1020,1920,0])
plt.plot(trunk_det[:,0], trunk_det[:,1])
plt.plot(trunk_rad2i[:,0], trunk_rad2i[:,1])
self = radar_tracker

target_trajectory = radar_tracker.live_radar_track[167]
source_trajectory = radar_tracker.live_detection_track[27]

    
plt.figure();plt.axis([0,1020,1920,0])
plt.plot(source_trajectory_overlap [:,0], source_trajectory_overlap [:,1])
plt.plot(target_trajectory_overlap_I [:,0], target_trajectory_overlap_I [:,1])    
#%%
貨車後面的機車
(168)
34

對面來的小黃
(169)

汽車
61
小學生
66

左側停汽車
1
10
11
12
13

中間角落停汽車
4
20
21
24
26
28
32
50
51
63
64

靠近的白色停汽車
3
16


27
(101.95488209194608, 1.1807112581722166)

51中間角落停汽車
(376.10576343536377, -0.03153317211241102)
59
(356.6205342610677, -0.012009658843119553)
62
(383.6821178089489, 0.34239168638306694)
44
(411.197995699369, 0.9466799730485107)
13左側停汽車
(741.6937296125624, 1.6474689936766538)
61汽車
(1095.351634979248, 1.3769321569897102)


32
(251.49552593809184, -178.35253100632326)
16
(261.87478404574927, -178.26798245060507)
34
(141.2166797344501, 49.608405127840285)
38
(493.03118896484375, -178.47315352183023)
43
(445.9081522623698, -179.6337305997517)


'''