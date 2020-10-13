# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:22:40 2020

@author: ystseng
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from CameraParameter import CameraParameter

#%%

class PerspectiveTransform:
    def __init__(self, pers_realxy, pers_video, latlong_realxy):
        ''' 
        args
            pers_realxy:    point list for realxy [(x1,y1), (x2,y2), ...]
            pers_video:     point list for video [(x1,y1), (x2,y2), ...]
            latlong_realxy: point list for realxy in Latitude and Longitude units [(lat1,long1), (lat2,long2), ...]
        '''
        xr, yr = 1, 1 #0.4, 1 #manual ration adjustment
        self.latlong_ratio = (110758.219/yr, 101751.561277/xr) # Latitude and longitude to meter in TW, ref: https://wywu.pixnet.net/blog/post/24805335
        self.M = cv2.getPerspectiveTransform(np.float32(pers_realxy), np.float32(pers_video)) #get perspctive matrix from realxy to video
        self.inv_M = cv2.getPerspectiveTransform(np.float32(pers_video), np.float32(pers_realxy)) #get perspctive matrix from realxy to video
        self.l2p, self.r2p = self.get_ratios(pers_realxy, pers_video, latlong_realxy) #meter to pixel ratio

    def get_ratios(self, pers_realxy, pers_video, latlong_realxy):
        '''compute meter to pixel ratios'''
        ratios = [ #get ratios of each 2 points (pixel/latlong)
            # here use [x] vs. [x-2] for taking longer pixel distance to improve accuracy
            np.divide(np.subtract(pers_realxy[x], pers_realxy[x-2]), np.subtract(latlong_realxy[x][::-1], latlong_realxy[x-2][::-1]))
            for x in range(int(len(pers_realxy)/2))
        ]      
        l2p = np.average(ratios, axis=0) #average ratios (pixel/latlong)
        return l2p, np.divide(l2p, self.latlong_ratio[::-1]) 

    def meter_to_pixel(self, pt):
        return tuple(np.multiply(pt, self.r2p))

    def latlong_to_pixel(self, pt):
        return tuple(np.multiply(pt[::-1], self.l2p))

    def project_to_pixel(self, pt):
        '''
        perform perspective projection (in pixel) 
        https://en.wikipedia.org/wiki/Transformation_matrix#Perspective_projection
        '''
        p = np.dot(self.M, (pt[0], pt[1], 1))
        p = np.divide(p, p[2])
        return tuple(p[0:2].astype(int))


class RSVTransformer(CameraParameter):
    '''雷達與影像與世界空間轉換模組
     
    設定雷達座標、影像座標與世界座標後，提供對應的計算轉換矩陣，以適合的方式來進行不同的空間投影。
    可以用空間轉換矩陣，來轉換多個點到對應座標空間的轉換。

    [Reference]
    1. https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=solvepnp
    2. N. Long, K. Wang, R. Cheng, K. Yang, and J. Bai, Proc. SPIE 10800, 108006 (2018).
    '''
    
    def __init__(self):
        self.mtx_W2I = np.array([])
        self.mtx_I2W = np.array([])
        self.mtx_W2R = np.array([])
        self.mtx_R2W = np.array([])
        self.mtx_R2I = np.array([])
        self.mtx_I2R = np.array([])
        self.world_points = np.array([], dtype = np.float32)
        self.radar_points = np.array([], dtype = np.float32)
        self.image_points = np.array([], dtype = np.float32)
        
    def set_world_points(self, world_points):
        '''輸入世界座標
        
        輸入世界座標，通常是以取樣點與原點的公尺數，GPS等座標。

        Parameters
        ----------
        world_points : np.array
            世界座標 : 2*N個樣本的np.array

        '''
        self.world_points = np.array(world_points, dtype=np.float32)
        
    def set_radar_points(self, radar_points):
        '''輸入雷達座標
        
        輸入雷達座標，通常是以以雷達本身為原點到取樣點的二維迪卡爾座標向量。非距離與方位角的組合。

        Parameters
        ----------
        radar_points : np.array
            雷達座標 : 2*N個樣本的np.array

        '''
        self.radar_points = np.array(radar_points, dtype=np.float32)
        
    def set_image_points(self, image_points):
        '''輸入影像座標
        
        輸入影像座標，通常是影像畫面上的x像素單位,y方向像素單位的多點組合。

        Parameters
        ----------
        image_points : np.array
            影像座標 : 2*N個樣本的np.array

        '''
        self.image_points = np.array(image_points, dtype=np.float32)
        
    def is_not_match(self, source_point_set, destination_point_set):
        '''確認點位長度
        
         用來判斷輸入的兩數列，取樣點數是否一樣長?

        Parameters
        ----------
        source_point_set : np.array
            來源數列 
            
        destination_point_set : np.array
            目標數列 
        '''
        return len(source_point_set) != len(destination_point_set)
    
    def is_empty(self,point_set):
        '''
        用來判斷是否有設定目標數列

        Parameters
        ----------
        point_set : np.array
            目標數列

        Returns
        -------
        TYPE bool
            若是沒有設定，則會輸出True

        '''
        return point_set.size == 0
            
    def calculate_radar_world_matrix(self):
        '''計算雷達座標轉世界座標矩陣
        
        以最小方差法來計算多點雷達座標與多點世界座標的轉換矩陣，以多點取樣來修正雷達與世界座標的誤差。
        其結果會存在物件內的mtx_W2R 與mtx_R2W 兩矩陣，可以被用來轉換雷達領域與世界領域的轉換。

        '''
        if self.is_empty(self.radar_points):
            print('Radar points not exist')
        if self.is_empty(self.world_points):
            print('world points not exist')
        if self.is_not_match(self.radar_points, self.world_points):
            print('radar and world sample is not match, please reset points')
            
        radar_x_array = self.radar_points[:,0]
        radar_y_array = self.radar_points[:,1]
        identity_matrix = np.array([1]*len(radar_x_array))
        world_Position_matrix = np.array([[world_point[0], world_point[1], 1] for world_point in self.world_points])
        
        transpose_column_x = np.linalg.lstsq(world_Position_matrix, radar_x_array, rcond=-1)[0]
        transpose_column_y = np.linalg.lstsq(world_Position_matrix, radar_y_array, rcond=-1)[0]
        transpose_column_offset = np.linalg.lstsq(world_Position_matrix, identity_matrix, rcond=-1)[0]
        transpose_matrix_world_to_radar = np.array([transpose_column_x.T, transpose_column_y.T, transpose_column_offset.T])
        transpose_matrix_radar_to_world = np.linalg.inv(transpose_matrix_world_to_radar)
        
        self.mtx_W2R = transpose_matrix_world_to_radar   
        self.mtx_R2W = transpose_matrix_radar_to_world 

    def calculate_world_to_image_matrix(self, ):
        '''計算世界座標轉到影像座標矩陣
        
        以opencv的solvePNP來計算世界座標轉影像座標需要的參數，需要先設定好相機校正參數。
        結果會存在物件內的mtx_W2I矩陣中。

        '''
        if self.is_empty(self.image_points):
            print('Image points not exist')
        if self.is_empty(self.world_points):
            print('world points not exist')
        if self.is_not_match(self.image_points, self.world_points):
            print('image and world sample is not match, please reset points')
            
        #TODO : 檢查是否有相機校正參數
        
        __worldPoints = np.array([[x[0], x[1], 1] for x in self.world_points])
        _, self.rotate_vector, self.transpose_vector = cv2.solvePnP(
            __worldPoints, self.image_points, 
            self.new_camera_matrix,
            self.dist, 
            flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        _rotate_matrix, _jac = cv2.Rodrigues(self.rotate_vector)
        rotate_transpose_matrix = np.column_stack((_rotate_matrix, self.transpose_vector))
        transpose_matrix_world_to_image = self.new_camera_matrix.dot(rotate_transpose_matrix)
        self.mtx_W2I = transpose_matrix_world_to_image
        
    def calculate_image_to_world_matrix(self, image_shape):   
        '''
        由於2D影像投射到3D中會有無窮解，需要額外增加參數，指定平面才有辦法解
        等哪天有有效解再建構。
        '''
        ##### XYZ
        # XYZ1 = np.array([[worldPoints[i,0],worldPoints[i,1],worldPoints[i,2],1]], dtype=np.float32)
        # XYZ1 = XYZ1.T
        # suv1 = P_mtx.dot(XYZ1)        
        # s = suv1[2,0]    
        # uv1 = suv1/(s*1.)
        # uv1 = uv1[0:2]
        ## cv2
        pass

        
    def save_parameter(self, path_save='RSV_metrix.npz'):
        '''儲存所有雷達轉換矩陣參數
        儲存本物件內的矩陣變數到指定路徑

        Parameters
        ----------
        path_save : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        np.savez(path_save, self.mtx_W2I, self.mtx_I2W, self.mtx_W2R, 
                            self.mtx_R2W, self.mtx_R2I, self.mtx_I2R, 
                            self.rotate_vector, self.transpose_vector, 
                            self.new_camera_matrix , self.dist )
    
    def load_parametre(self, path_load='RSV_metrix.npz'):
        '''讀取所有雷達轉換矩陣參數
        從指定路徑，載入本物件內的矩陣變數

        Parameters
        ----------
        path_save : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        mtxs = np.load(path_load)
        self.mtx_W2I, self.mtx_I2W, self.mtx_W2R, = mtxs['arr_0'], mtxs['arr_1'], mtxs['arr_2'],
        self.mtx_R2W, self.mtx_R2I, self.mtx_I2R, = mtxs['arr_3'], mtxs['arr_4'], mtxs['arr_5'], 
        self.rotate_vector, self.transpose_vector = mtxs['arr_6'], mtxs['arr_7'], 
        self.new_camera_matrix , self.dist = mtxs['arr_8'], mtxs['arr_9'], 



    
    def transepose_radar_to_world(self, radar_points):
        '''投影雷達點位到世界座標
        
        將多點雷達點位以雷達世界座標轉換矩陣做投影到世界座標。
        
        Parameters
        ----------
        radar_points : np.array
            雷達點位，為2*N的np.array。注意就算只有一點也必須以np.array([[x,y]])的方式輸入。

        Returns
        -------
        TYPE np.array
            已被投影至世界座標的點位。

        '''
        return np.array([self.mtx_R2W.dot([ii[0], ii[1],  1]) for ii in radar_points])[:,0:2]
    
    def transepose_world_to_radar(self, world_points):
        '''投影世界點位到雷達座標
        
        將多點世界點位以雷達世界座標轉換矩陣做投影到雷達座標。
        
        Parameters
        ----------
        world_points : np.array
            世界點位，為2*N的np.array。注意就算只有一點也必須以np.array([[x,y]])的方式輸入。

        Returns
        -------
        TYPE np.array
            已被投影至雷達座標的點位。

        '''
        return np.array([self.mtx_W2R.dot([ii[0], ii[1],  1]) for ii in world_points])[:,0:2]

    def transepose_world_to_image(self, world_points):
        '''投影世界點位到影像座標
        
        將多點世界點位以opencv的projectPoints做投影到影像座標。
        
        Parameters
        ----------
        world_points : np.array
            世界點位，為2*N的np.array。注意就算只有一點也必須以np.array([[x,y]])的方式輸入。

        Returns
        -------
        TYPE np.array
            已被投影至影像座標的點位。

        '''
        _worldPoints= np.array([[ii[0], ii[1], 1] for ii in world_points], dtype=np.float32)
        points_W2I, jco =cv2.projectPoints(_worldPoints, self.rotate_vector, self.transpose_vector , self.new_camera_matrix , self.dist)     
        return points_W2I[:,0]

    def transepose_radar_to_image(self, radar_points):
        '''投影雷達點位到影像座標
        
        將多點雷達點位以投影到世界座標，再投影到影像座標兩步驟來做轉換。
        
        Parameters
        ----------
        radar_points : np.array
            世界點位，為2*N的np.array。注意就算只有一點也必須以np.array([[x,y]])的方式輸入。

        Returns
        -------
        TYPE np.array
            已被投影至影像座標的點位。

        '''
        _radar_project_to_world = self.transepose_radar_to_world(radar_points)
        return self.transepose_world_to_image(_radar_project_to_world )
    
    def transepose_image_to_world(self, image_points):
        '''
        由於2D影像投射到3D中會有無窮解，需要額外增加參數，指定平面才有辦法解
        等哪天有有效解再建構。
        '''
        print('It needs scale matrix')
        pass

    def transepose_image_to_radar(self, image_points):
        '''
        由於2D影像投射到3D中會有無窮解，需要額外增加參數，指定平面才有辦法解
        等哪天有有效解再建構。
        '''
        print('It needs scale matrix')
        pass
    
 
#%%


#%%
def main():
    #%%
    cap = cv2.VideoCapture(r'C:\\GitHub\\109_RadarFusion\\Dataset\\雷達資料0505\\0505b\\2020-05-05_15-20-35.avi')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200 )
    ret, frame = cap.read()     
    
    points = [[[10,0],  [570,954]],
              [[20,0],  [340,1010]],
              [[30,0],  [247,1030]],
              [[40,0],  [203,1043]],
              [[50,0],  [177,1050]],
              [[60,0],  [155,1052]],
              [[70,0],  [147,1054]],
              [[10,6],  [462,359]],
              [[20,6],  [282,659]],
              [[30,6],  [213,784]],
              [[40,6],  [177,855]],
              [[50,6],  [154,896]],
              [[60,6],  [140,923]],
              [[70,6],  [130,944]],
              [[10,10], [396,23]],
              [[20,10], [250,479]],
              [[30,10], [197,642]],
              [[40,10], [167,740]],
              [[50,10], [145,806]],
              [[60,10], [132,836]],
              [[70,10], [125,865]],]
    
    image_points=np.array(points)[:,1,:]
    world_points=np.array(points)[:,0,:]
    
    rt = RSVTransformer()
    rt.set_image_points(image_points)
    rt.set_world_points(world_points)
    rt.load_camera_parameter_by_path(r'C:\\GitHub\\109_RadarFusion\\panasonic_camera\\')
    rt.set_new_camera_matrix(np.array([[1805.41,	   0,	924.743],
                                   [      0,	1100,	539.679],
                                   [      0,	   0,	      1]]))
    rt.calculate_world_to_image_matrix()
    rt.transepose_world_to_image(np.array([[10,10]]))
    world2image_points = rt.transepose_world_to_image(world_points)

    cmap = plt.get_cmap('hsv') # 'gist_rainbow','Paired'
    rgb = [cmap(int(x)) for x in np.linspace(0,255,len(world_points))]    
    plt.figure()
    plt.title('Real world coordination to Image coordination, \no:image, x:world project to image')
    plt.imshow(frame[:,:,::-1])
    for ii, point in enumerate(world2image_points):
        plt.plot(point[1],point[0],'x',color=rgb[ii],markersize=8,markeredgewidth=2)
        # plt.text(point[0],point[1], '%.02f'%s_arr[ii], fontsize=15)
    for ii, point in enumerate(image_points):
        plt.plot(point[1],point[0],'o',color=rgb[ii],markersize=6)
    plt.show()
        
    #%%
    world_points = np.array([[35,2],
                            [50,2],
                            [68,10],
                            [50,10],
                            [28,10],])
    radar_points = np.array([[40.5, -0.6],
                             [63.3, -3.0],
                             [64.6,  3.0],
                             [48.0,  4.7],
                             [27.4,  7.9],])
    rt.set_radar_points(radar_points)
    rt.set_world_points(world_points)
    rt.calculate_radar_world_matrix()
    rt.transepose_radar_to_world(radar_points)
    radar2image_points = rt.transepose_radar_to_image(radar_points)
    
    plt.figure()
    plt.title('Radar Porject to image coordination, \n^:radar points')
    plt.imshow(frame[:,:,::-1])
    for ii, point in enumerate(radar2image_points ):
        plt.plot(point[1],point[0],'^',color=rgb[ii],markersize=6)

    #%%
    radar2world_points = rt.transepose_radar_to_world(radar_points)
    world2image_points = rt.transepose_world_to_radar(world_points)
    
    plt.figure()
    plt.title('World Radar adjust , \n x:radar domain, o:world domain')
    plt.plot(-world_points[:,1],world_points[:,0],'ro',markersize=10 )
    plt.plot(-radar_points[:,1],radar_points[:,0],'bx',markersize =10 )
    plt.plot(-radar2world_points[:,1],radar2world_points[:,0],'go',markersize=5 )
    plt.plot(-world2image_points[:,1],world2image_points[:,0],'cx',markersize=5 )
    plt.axis([-30,30,20,80])

#%%
if __name__ == '__main__':
    main()