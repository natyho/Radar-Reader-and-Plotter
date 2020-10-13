#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:13:35 2020

@author: ystseng
"""


import sys, os
# sys.path.append('event/')
import tkinter as tk
from time import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from time import sleep


import matplotlib.patches as patches

from RSV_Capture import RadarCapture, IPcamCapture
from RSV_Transformer import RSVTransformer

#%%
class TKInputDialog:
    #ref: https://www.python-course.eu/tkinter_entry_widgets.php
    def __init__(self, labels:list=['x', 'y'], defaults:list=[], title=None, win_size=(200, 150)):
        self.master = tk.Tk()
        self.entries = {}
        self.ret = None
        if title:
            tk.Label(self.master, text=title).pack(side=tk.TOP)

        #create input labels and entries
        for i, label in enumerate(labels):
            row = tk.Frame(self.master)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            lab = tk.Label(row, width=5, text=label, anchor='w')
            lab.pack(side=tk.LEFT)
            ent = tk.Entry(row, width=3)
            ent.insert(0, defaults[i] if any(defaults) else '') #fill default value
            ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.entries[label] = ent

        #confirm actions
        tk.Button(self.master, text='Ok', command=self.ok).pack(side=tk.LEFT, padx=5, pady=5) #press OK button
        self.master.protocol('WM_DELETE_WINDOW', self.ok) #close window https://stackoverflow.com/a/111160/10373104
        self.master.bind('<Return>', self.ok) #press Enter https://stackoverflow.com/a/16996475/10373104

        #adjust window position & size https://stackoverflow.com/a/14910894/10373104
        w, h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry('%dx%d+%d+%d' % (win_size[0], win_size[1], int(w/2), int(h/2)))

        self.master.focus_force() #move focus to this window https://stackoverflow.com/a/22751955/10373104
        self.master.mainloop()

    def ok(self, event=None):
        def to_float(x):
            try:
                return float(x)
            except ValueError:
                return x
        self.master.quit()
        # self.ret = {key: float(val.get()) for key, val in self.entries.items()}
        self.ret = [to_float(val.get()) for key, val in self.entries.items()]
    
    def get(self):
        self.master.withdraw() #close Tk window
        return self.ret


class TKSrollDownDialog:
    #ref: https://stackoverflow.com/a/45442534/10373104
    def __init__(self, options:list=[], title=None, win_size=(150, 100)):
        self.master = tk.Tk()

        if title:
            tk.Label(self.master, text=title).pack(side=tk.TOP)

        # setup scroll down menu
        self.ret = tk.StringVar(self.master)
        self.ret.set(options[0]) # default value
        w = tk.OptionMenu(self.master, self.ret, *options).pack()

        #confirm actions
        tk.Button(self.master, text='Ok', command=self.ok).pack(side=tk.BOTTOM, padx=5, pady=5) #press OK button
        self.master.protocol('WM_DELETE_WINDOW', self.ok) #close window https://stackoverflow.com/a/111160/10373104
        self.master.bind('<Return>', self.ok) #press Enter https://stackoverflow.com/a/16996475/10373104

        #adjust window position & size https://stackoverflow.com/a/14910894/10373104
        w, h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry('%dx%d+%d+%d' % (win_size[0], win_size[1], int(w/2), int(h/2)))

        self.master.focus_force() #move focus to this window https://stackoverflow.com/a/22751955/10373104
        self.master.mainloop()

    def ok(self, event=None):
        def to_float(x):
            try:
                return float(x)
            except ValueError:
                return x
        self.master.quit()
        # self.ret = 
        # self.ret = {key: float(val.get()) for key, val in self.entries.items()}
        # self.ret = [to_float(val.get()) for key, val in self.entries.items()]
    
    def get(self):
        self.master.withdraw() #close Tk window
        return self.ret.get()


class MousePointsReactor:
    '''return desired mouse click points on frame get_mouse_points'''
    #https://stackoverflow.com/questions/33370613/using-mouse-to-plot-points-on-image-in-python-with-matplotlib
    def __init__(self, img, num, labels:list=None, defaults:list=None):
        self.pts = {} #final results
        self.circles = [] #double click positions
        self.texts = [] #show text for circles
        self.line_pts = [] #alignment lines [[(x1, y1), (x2, y2)], ...]
        self.newline = [] #for drawing a line
        self.l = None #temporary line patch
        self.img = img
        self.num = num
        self.ax = None #pyplot canvas
        self.labels = labels
        self.defaults = defaults
        self.str_defaults = [str(x) for x in defaults]
        self.lock = False #lock line drawing
        self.leave = False
        self.loop = True
        master = tk.Tk() #dummy tk root window
        master.withdraw() #remove blank window https://stackoverflow.com/a/17280890/10373104

    def onClick(self, event):
        if not event.xdata or not event.ydata: #click outside image
            return
        pos = (event.xdata, event.ydata)
        if event.button == MouseButton.LEFT:
            #check click-on-circle event
            click_on_circle = False
            for i_c, c in enumerate(self.circles):
                ## check every exist circule 
                contains, attrd = c.contains(event)
                if contains:
                    click_on_circle = True
                    #get values from popup
                    
                    list_unpicked_item = [iy for iy,y in enumerate(self.defaults) if str(y) not in [str(x) for x in self.pts.values()]]
                    list_picked_item = [iy for iy,y in enumerate(self.defaults) if str(y) in [str(x) for x in self.pts.values()]]
                    list_show = [self.str_defaults[x] for x in list_unpicked_item ] +['--------']+[self.str_defaults[x] for x in list_picked_item ]
                    list_show_result = list_unpicked_item+ [999]+list_picked_item 
                    idx = list_show.index(TKSrollDownDialog(list_show).get())
                    
                    if list_show_result [idx]!=999:
                        # if not choose '--------'
                        # if choose pickuped item, remove text 
                        self.pts.update({c.center:self.defaults[list_show_result [idx]]})
                        if self.str_defaults[list_show_result[idx]] in [x.get_text() for x in self.texts]:
                            drop_index = [x.get_text() for x in self.texts].index(self.str_defaults[list_show_result[idx]])
                            self.texts.pop(drop_index).remove()
                        # draw the new text
                        t = self.ax.text(c.center[0]+10, c.center[1]-10, self.str_defaults[list_show_result[idx]], color='yellow')
                        self.texts.append(t)
                        
                    else:
                        # if choose '--------', remove node
                        c=self.circles[0]
                        if (c.center[0]+10,c.center[1]-10) in [x.get_position() for x in self.texts]:
                            # remove text
                            drop_index =[x.get_position() for x in self.texts].index((c.center[0]+10,c.center[1]-10))
                            self.texts.pop(drop_index).remove()
                            # remove point
                            del(self.pts[c.center])
                    break                      

            #not click on circle -> draw lines
            if not click_on_circle and not self.lock:
                if not self.newline: #add line start point
                    # print('single1')
                    self.newline = [pos]
                else: #add line end point
                    # print('single2')
                    self.newline.append(pos)
                    self.line_pts.append(self.newline)
                    self.newline = []
                    self.l = None #reset patch (unbound to the finished line)
                    self.show_intersections()
            click_on_circle = False #reset
        elif event.button == MouseButton.RIGHT and not self.lock:
            # print('remove')
            def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
                a=lineY2-lineY1
                b=lineX1-lineX2
                c=lineX2*lineY1-lineX1*lineY2
                dis=(np.abs(a*pointX+b*pointY+c))/(np.sqrt(a*a+b*b))
                return dis                 
            try:
                list_dis = []
                for lines in self.line_pts:
                    list_dis.append(getDis(pos[0],pos[1], lines[0][0],lines[0][1],lines[1][0],lines[1][1]))
                index_drop = np.argmin(list_dis)
                
                del(self.line_pts[index_drop])
                self.ax.lines[index_drop].remove()
                self.show_intersections()
            except:
                pass

        self.ax.figure.canvas.draw_idle() #update canvas            

    def onMove(self, event):
        if not event.xdata or not event.ydata or self.lock: #click outside image
            return
        pos = (event.xdata, event.ydata)
        if self.newline: #has start point
            try:
                self.l.remove() #or self.ax.lines[0].remove() or self.ax.lines.remove(self.l)
            except:
                pass
            #Line2D https://stackoverflow.com/a/36488527/10373104
            self.l = Line2D([self.newline[0][0], pos[0]], [self.newline[0][1], pos[1]], color='red')
            self.ax.add_line(self.l)
        self.ax.figure.canvas.draw_idle() #update canvas                

    def onKyePress(self, event):
        self.lock = self.lock ^ (event.key in ['x', 'X']) #update lock status
        if self.lock:
            plt.title('Line mode: Left to draw, Right to remove.\nSet mode: cleck to select nodes.\npress x to switch modes.\npress L or Q to leave\nCurrent at Set Mode')
        else:
            plt.title('Line mode: Left to draw, Right to remove.\nSet mode: cleck to select nodes.\npress x to switch modes.\npress L or Q to leave\nCurrent at Line Mode')
        self.leave = event.key in ['l', 'L', 'Q','q']
    
    def start(self):
        self.loop = True
        self.leave = False
        fig, self.ax = plt.subplots(1)
        plt.imshow(self.img[:,:,::-1]) #or self.ax.imshow(self.img[:,:,::-1])
        # cid1 = fig.canvas.mpl_connect('button_press_event', self.onClick)
        # cid2 = fig.canvas.mpl_connect('motion_notify_event', self.onMove)
        # cid3 = fig.canvas.mpl_connect('key_press_event', self.onKyePress)
        fig.canvas.mpl_connect('button_press_event', self.onClick)
        fig.canvas.mpl_connect('motion_notify_event', self.onMove)
        fig.canvas.mpl_connect('key_press_event', self.onKyePress)
        plt.title('Line mode: Left to draw, Right to remove.\nSet mode: cleck to select nodes.\npress x to switch modes.\npress L or Q to leave\nCurrent at Line Mode')
        plt.show()
        # auto close version
        plt.show(block=False) #https://github.com/matplotlib/matplotlib/issues/8560/#issuecomment-397254641
        while (len(self.pts) < self.num) & self.loop :
            if self.leave:
                self.loop = False
            if cv2.waitKey(1) in [ord('q'), ord('Q'), 27]:
                break
        sleep(1)
        plt.close(fig)
        print('lines = ', self.line_pts)
        return self.pts
    
    def get_pts(self):
        return self.pts
    def get_inline_intersections(self):
        '''get inline intersection points from drawed straight lines'''
        def is_within(x, p1, p2): #p1, p2=(x1, y1), (x2, y2)
            return (p1[0]<=x[0]<=p2[0] or p1[0]>=x[0]>=p2[0]) and (p1[1]<=x[1]<=p2[1] or p1[1]>=x[1]>=p2[1])
        intersections = []
        for i, p1 in enumerate(self.line_pts): #p1 = [(a, b), (c, d)]
            #formulate as px + qy = r
            m1 = (p1[0][1] - p1[1][1], p1[1][0] - p1[0][0]) #(p, q) = (b-d, c-a)
            b1 = p1[1][0]*p1[0][1] - p1[0][0]*p1[1][1] #cb-ad

            for j, p2 in enumerate(self.line_pts[i+1:]):
                #formulate as px + qy = r
                m2 = (p2[0][1] - p2[1][1], p2[1][0] - p2[0][0]) #(p, q) = (b-d, c-a)
                b2 = p2[1][0]*p2[0][1] - p2[0][0]*p2[1][1] #cb-ad

                #find intersection point
                #reference: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/87358/
                try:
                    r = np.linalg.solve(np.mat([m1, m2]), np.mat([b1, b2]).T)
                    if is_within(r, p1[0], p1[1]) and is_within(r, p2[0], p2[1]):
                        intersections.append((int(r[0][0]), int(r[1][0])))
                except np.linalg.LinAlgError: #no intersection
                    pass
        return intersections

    def show_intersections(self):
        '''show inlin intersection points on canvas and storing'''
        intersections = self.get_inline_intersections()        
        # clear original patches
        for i in range(len(self.circles)):
            self.circles.pop(-1).remove()
        for i in range(len(self.texts)):
            self.texts.pop(-1).remove()
        self.circles = []
        self.texts = []
        self.pts = {}
        # draw new circles
        for p in intersections:            
            c = patches.Circle(p, 7, color='blue', zorder=100) #draw a circle on top
            self.ax.add_patch(c)
            self.circles.append(c)


#%%
def set_SV_transformer(rt, src, radar=None, flag_reset=False):
    
    #是否為離線模式    
    offline = not src.startswith('rtsp')
    vout = src.startswith('rtsp')

    ## 連接攝影機
    ipcam = IPcamCapture(src, offline=offline, start_frame=0)
    ipcam.modify_time_drify(second=2, microsecond=200000)

    # 啟動子執行緒
    ipcam.start(write=vout)
    # 暫停1秒，確保影像已經填充
    sleep(0.5)
    img, _ = ipcam.get_frame()
    
    ''' UI for world to video'''
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
    
    image_points = np.array(points)[:,1,:]
    world_points = np.array(points)[:,0,:]

    # mouse configuring or load existance
    if os.path.exists('points_image.npz') & ~flag_reset:
        print('read npy')
        points = np.load('points_image.npz')
        image_points, world_points = points['arr_0'], points['arr_1']
    else:
        print('redraw point')
        get_points = MousePointsReactor(img, len(world_points), ['x', 'y'], world_points)
        points = get_points.start()

        world_points = np.array(list(points.values()))
        image_points = np.array([np.array(x) for x in points.keys()])
        np.savez('points_image.npz', image_points, world_points)
    
    #set1
    from scipy.optimize import minimize
    rt.set_image_points(image_points)
    rt.set_world_points(world_points)
    rt.load_camera_parameter_by_path(os.path.join(os.getcwd(), 'event'))#r'C:\\GitHub\\109_RadarFusion\\panasonic_camera\\')
    def opt_camera_parameter(f):
        # print(r'{f[0]} {f[1]}')
        rt.set_new_camera_matrix(np.array([[f[0],	     0,	f[2]],
                                           [   0,	f[1],	f[3]],
                                           [   0,	 0,	      1]],dtype=np.float64))
        rt.calculate_world_to_image_matrix()
        pixel_point = rt.transepose_world_to_image(world_points)
        return np.sum(np.abs(pixel_point -image_points))
    
    res = minimize(opt_camera_parameter, [1300,1500,1000,500], method='Nelder-Mead', tol=1e-6)
    opt_camera_parameter(res.x)
    pixel_point = rt.transepose_world_to_image(world_points)
    plt.figure()
    plt.imshow( img[:,:,::-1])
    plt.plot(pixel_point[:,0], pixel_point[:,1], 'ro' )
    plt.plot(image_points[:,0], image_points[:,1], 'bx' )
    return rt


def set_RS_transformer(rt, radar, flag_reset=False):
    meter_pixel = 12    
    radar_range_y = 30
    radar_range_x = 80
    radar_location = ((radar_range_x-5)*meter_pixel, (radar_range_y-15)*meter_pixel)
    def pixel_radar(x, y, theta=0, dxy=[0,0]):
        '''radar to world pixel transformer'''
        th=theta*np.pi/180
        M=np.array([[np.cos(th), - np.sin(th)],[np.sin(th), np.cos(th)]])
        v=np.array([[y],[x]])
        v_r = np.dot(M,v)    
        y= v_r[0]-dxy[0]
        x= v_r[1]+dxy[1]
        return (int(radar_location[1]-y*meter_pixel), int(radar_location[0]-x*meter_pixel))
    
    def draw_birdview_basemap(dict_target={}, dict_road={}):
        '''world pixel basemap'''
        ## plot radar background 
        radar_plane = np.zeros([radar_range_x*meter_pixel, radar_range_y*meter_pixel, 3], dtype = np.uint8)*10
    
        ## plot target region with 3 meter
        for distance in range(10):
            # distance=0
            cv2.line(radar_plane, pixel_radar(distance*10, -15), pixel_radar(distance*10, 15), (100, 100, 100), 1)        
        
        for target_key in dict_road.values():
            cv2.line(radar_plane, pixel_radar(target_key[0][0], target_key[0][1]), pixel_radar(target_key[1][0], target_key[1][1]), (150, 150, 150), 3)
        
        for target_key in dict_target.values():
            cv2.circle(radar_plane , pixel_radar(target_key[0], target_key[1]), int(3*meter_pixel), (0,0,255),2)        
            
        # cv2.imshow('radar plane', cv2.resize(radar_plane ,(0,0),fx=1,fy=1) )
        return radar_plane 
    
    class SaveRadarUIPoint():
        '''Radar Point sets'''
        def __init__(self, num_points, dict_target):
            self.points=[]
            self.num_points = num_points
            self.dict_target = dict_target
        def append(self,point):
            self.points.append(point)
            
        def start(self):
            points = TKInputDialog(
                labels=['x:%s\ny:%s'%(self.dict_target[x][0], self.dict_target[x][1]) for x in self.dict_target], 
                defaults=[], 
                title='Enter the radar sample data', 
                win_size=(240, 50 * self.num_points)
                ).get()
            if points != '':
                for i_p,point in enumerate(points):
                    if point != '':
                        self.points.append({ i_p:np.array([float(pp) for pp in point.lstrip().split(' ')]) })
                    
        def get_RW_points(self):
            radar_points = np.array([list(x.values())[0] for x in self.points])
            world_points = np.array([self.dict_target[list(x.keys())[0]] for x in self.points])
            
            return radar_points, world_points 
    '''set radar to world point sets'''
    if os.path.exists('points_radar.npz') & ~flag_reset:        
        print('read npy')
        points = np.load('points_radar.npz')
        radar_points, world_points = points['arr_0'], points['arr_1']
    else:      
        world_points = np.array([[40,2],
                                [55,2],
                                [68,10],
                                [50,10],
                                [28,10],])
        radar_points = np.array([[40.5, -0.6],
                                 [63.3, -3.0],
                                 [64.6,  3.0],
                                 [48.0,  4.7],
                                 [27.4,  7.9],])
        num_points = TKInputDialog(
            labels=['N = '], 
            defaults=[], 
            title='Set total number of target regions', 
            win_size=(200, 50)
        ).get()[0]
        try:
            num_points = int(num_points)
        except:
            num_points =5
            print('Enter Number is not a interger, set region to 5')
    
        if num_points <=5:
            default_pair = ['%d %d'%(str_default[0],str_default[1]) for str_default in world_points[:num_points]]
        else:
            default_pair = [list(x) for x in world_points]+['']*(num_points-5)
            
        str_target_centers = TKInputDialog(
            labels=['R %d'%num for num in range(num_points)], 
            defaults=default_pair , 
            title='Set target regions centers', 
            win_size=(240, 50*num_points)
        ).get()  
    
        dict_target = dict(enumerate([[float(point)for point in strpair.split(' ')]for strpair in str_target_centers if strpair !='']))
        
        dict_road ={ '1':[(0,0), (80,0)],
                     '2':[(0,6), (80,6)],
                     '3':[(0,12),(80,12)],
                     # '3':[(0,12),(80,12)]
            }
        save_radar_point = SaveRadarUIPoint(num_points, dict_target)
        radar_adjust_rotation = 0
        radar_drift_position = [0,0]
        last_radar_objects ={}
        list_save_object =[]
        list_slow_points = []
        list_fast_points = []    
        
        ## A. start radar
        radar_start_time = system_start_time = None
        radar.get_info()
        radar.start_parse()

        radar_start_time = None
        while not radar_start_time:
            radar_start_time = radar.start_worktime
        system_start_time = time()*1000
        # frame_i  =0##Debug
        # radar_start_time = datetime.strptime(radar_data[0]['time'], "%Y-%m-%d %H:%M:%S.%f").timestamp()*1000##Debug
        difference_time_radar_system = system_start_time -radar_start_time 
        size_text = 0.5
        flag_loop = True
        flag_first = False
    
    

        while flag_loop:
    
            ## a. get radar objects 
            search_radar_time = time() - difference_time_radar_system 
            current_radar_objects = radar.get_current_object(search_radar_time , nearest=True)
            # current_radar_objects = radar_data[frame_i ]##Debug
            # frame_i +=1##Debug
            ## b. get radar plant background
            if not current_radar_objects:
                continue
            radar_plane_raw = draw_birdview_basemap(dict_target, dict_road)
            radar_plane = radar_plane_raw.copy()
            cv2.putText(radar_plane, current_radar_objects['time'], (10,20), cv2.FONT_HERSHEY_SIMPLEX, size_text, (0,0,255),2,cv2.LINE_4)
            # cv2.imshow('radar plane', cv2.resize(radar_plane ,(0,0),fx=1,fy=1) )
        
            ## c. draw radar objects
            for radar_key in [obj for obj in current_radar_objects.keys() if obj !='time']:
                # radar_key ='02'
                radar_object = current_radar_objects[radar_key]
                cv2.circle(radar_plane, pixel_radar(radar_object['x'], radar_object['y'],radar_adjust_rotation,radar_drift_position), int(radar_object['length']*1.5), (0,255,0),-1)
                cv2.putText(radar_plane, '%s, %.02f, %.02f'%(radar_object['oid'], radar_object['x'], radar_object['y']), 
                            pixel_radar(radar_object['x'], radar_object['y'],radar_adjust_rotation,radar_drift_position ), 
                            cv2.FONT_HERSHEY_SIMPLEX, size_text, (0,0,255),1,cv2.LINE_AA)
            
            '''
            last_radar_objects=radar_data[155]## Debug
            current_radar_objects=radar_data[156]## Debug
            '''
            if flag_first:
                ## d. find dissappear object, and add to temp record
                old_radar_IDs = {last_radar_objects[object_ID]['oid']:object_ID  for object_ID in last_radar_objects.keys() if object_ID != 'time' }
                new_radar_IDs = {current_radar_objects[object_ID]['oid']:object_ID  for object_ID in current_radar_objects.keys() if object_ID != 'time' }
                for old_radar_ID in old_radar_IDs:
                    if old_radar_ID not in new_radar_IDs:
                        list_save_object.append(last_radar_objects [old_radar_IDs[old_radar_ID]])
    
                ## e.shwo slow objects     
                for current_radar_key in [obj for obj in current_radar_objects.keys() if obj !='time']:
                    # get position two points
                    radar_object = current_radar_objects[current_radar_key ]
                    if current_radar_key in last_radar_objects.keys():
                        radar_object_last = last_radar_objects [current_radar_key ] 
                    else:
                        continue
                    
                    # check speed type
                    speed_up_limit = 30*000/60/60 ## km speed * radar fps
                    speed_low_limit = 7*1000/60/60*0.075 ## km speed * radar fps
                    if np.linalg.norm([(radar_object['x'] -radar_object_last['x']), (radar_object['y'] -radar_object_last['y'])]) < speed_low_limit :
                        list_slow_points.append((radar_object['x'],radar_object['y']))
                    elif np.linalg.norm([(radar_object['x'] -radar_object_last['x']), (radar_object['y'] -radar_object_last['y'])]) > speed_up_limit :
                        list_fast_points.append((radar_object['x'],radar_object['y']))
    
            else:            
                flag_first = True
            last_radar_objects = current_radar_objects.copy()
    
            ## f.show slow point (purple)
            list_slow_points = list_slow_points[-200:]
            for pp in list_slow_points:
                cv2.circle(radar_plane, pixel_radar(pp[0], pp[1],radar_adjust_rotation, radar_drift_position), int(3), (150,0,255),-1)
                
            ## g.show fast point (Aqua blue)
            list_fast_points = list_fast_points[-200:]
            for pp in list_fast_points:
                cv2.circle(radar_plane, pixel_radar(pp[0], pp[1],radar_adjust_rotation, radar_drift_position), int(1), (255,200,0),-1)
            
            ## h.keyboard activation
            keyevent = cv2.waitKey(20)
            if keyevent in [ord('a'), ord('A'), ord('D'),ord('d'), 27, ord('w'),ord('W'), ord('E'),ord('e'),
                        ord('i'),ord('k'),ord('j'),ord('l'),ord('I'),ord('K'),ord('J'),ord('L'),ord('s'),ord('S')]:
                if keyevent==27:
                    flag_loop=False
                
                if keyevent in [ord('D'),ord('d')]:
                    list_save_object =[]
                    list_slow_points = []
                    list_fast_points = []
                    
                if keyevent in [ord('A'),ord('a')]:
                    for radar_key in [obj for obj in current_radar_objects.keys() if obj !='time']:
                        list_save_object.append(current_radar_objects[radar_key ])
                        
                if keyevent in [ord('W'),ord('w')]:
                    # print('left')s
                    radar_adjust_rotation -=0.1
                if keyevent in [ord('E'),ord('e')]:
                    # print('right')
                    radar_adjust_rotation +=0.1
                if keyevent in [ord('I'),ord('i')]:
                    radar_drift_position[1]=radar_drift_position[1]+0.1
                if keyevent in [ord('K'),ord('k')]:
                    radar_drift_position[1]=radar_drift_position[1]-0.1
                if keyevent in [ord('J'),ord('j')]:
                    radar_drift_position[0]=radar_drift_position[0]-0.1
                if keyevent in [ord('L'),ord('l')]:
                    radar_drift_position[0]=radar_drift_position[0]+0.1
                if keyevent in [ord('s'),ord('S')]:
                    save_radar_point.start()
                    
            # print('list_slow_points: %d'%len(list_slow_points))
            # print('list_fast_points: %d'%len(list_fast_points))
            # print('list_save_object: %d'%len(list_save_object))
            ## i.draw save nodes (yellow)
            for radar_object in list_save_object[-20:]:
                    # radar_key ='02'
                    # radar_object = saveframe[radar_key]
                    cv2.circle(radar_plane, pixel_radar(radar_object['x'], radar_object['y'],radar_adjust_rotation ), int(radar_object['length']*1.5), (0,255,255),-1)
                    cv2.putText(radar_plane, '%s, %.02f, %.02f'%(radar_object['oid'], radar_object['x'], radar_object['y']), pixel_radar(radar_object['x'], radar_object['y'],radar_adjust_rotation ), cv2.FONT_HERSHEY_SIMPLEX, size_text, (0,255,255),1,cv2.LINE_AA)
            ## j.draw radar position
            cv2.circle(radar_plane , pixel_radar(0,0,radar_adjust_rotation,radar_drift_position), int(10), (0,0,255),-1) 
            cv2.line(radar_plane, pixel_radar(0,0,radar_adjust_rotation,radar_drift_position), pixel_radar(80,0,radar_adjust_rotation,radar_drift_position), (0,200,255),4)
        
            cv2.imshow('radar plane', cv2.resize(radar_plane,(0,0) ,fx=1,fy=1))#600,1020

        print('redraw point')
        radar_points, world_points = save_radar_point.get_RW_points()
        np.savez('points_radar.npz', radar_points, world_points)

    rt.set_radar_points(radar_points)
    rt.set_world_points(world_points)
    rt.calculate_radar_world_matrix()
    
    rt.save_parameter()
    return rt


#%%
def main_test_MousePointsReactor():
    ret, img = cv2.VideoCapture(r'C:\Dropbox\event_engine\test\DJI_0002_15FPSa\DJI_0002_15FPSa.m4v').read()
    pts = MousePointsReactor(img, 2, ['x', 'y'], [[1,1],[2,2],[3,3],[4,4],[5,5]]).start()#, [[1],[2],[3],[4],[5]]).start()
    # pts = TKInputDialog(['a', 'b', 'c', 'd', 'e'], title='test', win_size=(200, 50*5)).get()
    print('pts =', pts)
    
def main(src, COM='COM13', filepath=None):
    '''用來設置雷達錨點的主程式

    Parameters
    ----------
    src : string
        串流路徑或者是影片路徑
    COM : string, optional
        雷達的comport. The default is 'COM13'.
    filepath : string, optional
        log的存放位置. The default is None.

    Returns
    -------
    None.

    src=r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cc\2020-06-04_17-03-11,698371.mp4'
    COM='COM13'
    filepath= r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cc\radar.log'
    '''

    ## 建立投影矩陣
    rt_set = RSVTransformer()
    rt_set.load_camera_parameter_by_path(os.path.join(os.getcwd(), 'event'))
    
    ## A. 設置世界投影影像矩陣
    rt_set = set_SV_transformer(rt_set, src, flag_reset=False)


    ## B.設置雷達頭影視界矩陣
    filepath = None if src.startswith('rtsp') else filepath
    rt_set = set_RS_transformer(rt_set, RadarCapture(COM, filepath=filepath), flag_reset=True)
    
    ## C.儲存投影矩陣結果
    rt_set.save_parameter(path_save='RSV_metrix.npz')
    
    #### 測試結果
    ## A.讀取轉至矩陣
    rt = RSVTransformer()
    rt.load_parametre(path_load='RSV_metrix.npz')
    rt.mtx_W2I

    ## B. 連接攝影機
    _, img = cv2.VideoCapture(src).read()
    plt.figure();plt.imshow(img[:,:,::-1])
    world_paths = np.array([[np.linspace(10,80,200), [0]*200],
                            [np.linspace(10,80,200),[6]*200],
                            [np.linspace(10,80,200),[10]*200,]])    
    for x in range(200):
        
        pp = rt.transepose_world_to_image([world_paths[0,:,x]])[0]
        cv2.circle(img , (int(pp[0]), int(pp[1])), int(2), (0,0,255),-1)
        
        pp = rt.transepose_world_to_image([world_paths[1,:,x]])[0]
        cv2.circle(img , (int(pp[0]), int(pp[1])), int(2), (0,255,0),-1)
        
        pp = rt.transepose_world_to_image([world_paths[2,:,x]])[0]
        cv2.circle(img , (int(pp[0]), int(pp[1])), int(2), (255,0,0),-1)
        
    plt.imshow(img[:,:,::-1])
        
#%%    
if __name__ == '__main__':
        dict_carmer = {
            'KL' : "rtsp://keelung:keelung@nmsi@60.251.176.43/onvif-media/media.amp?streamprofile=Profile1",
            'KH' : "rtsp://demo:demoIII@221.120.43.26:554/stream0",
            'IP' : 'rtsp://192.168.0.100/media.amp?streamprofile=Profile1',
            'LL' : r'F:\GDrive\III_backup\[Work]202006\0604雷達資料\0604cc\2020-06-04_17-03-11,698371.mp4',
            'LLt' : r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cc\2020-06-04_17-03-11,698371.mp4'
        }
        filepath = r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cc\radar.log'
        main(dict_carmer['KL'], COM='COM13', filepath=filepath)
        sys.exit()
