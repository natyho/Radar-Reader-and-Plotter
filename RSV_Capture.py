# -*- coding: utf-8 -*-
"""Reading SmartMicro Radar data by pySerial
Authors: 
Chia-Hsin Chan terry0201@gmail.com

References:
[ch_42] Smart Micro UMRR-0C 泛用交通雷達/車輛偵測器 資料協議說明 Version1
[en_42] Data Communication UMRR0C Traffic Management June 5, 2019
"""
import sys, os
sys.path.append('event/')
import json
import serial
import serial.tools.list_ports
import cv2
import numpy as np
from threading import Thread
from time import sleep, time
from datetime import datetime, timedelta
from collections import deque
from copy import deepcopy

#from event_engine project
from utils.dataloader import LoadVideo
from event import logger
from event.util import FPSCounter

Log = logger.GetLogger("radar", "radar", 'DEBUG').initial_logger()

class RadarCapture:
    '''
    args:
        port: COM#
        filepath: offline radar log file path
        radar_type: 42/30
        protocal: RS485/Ethernet
        bufflen: object buffer max length
    '''
    def __init__(self, port='COM7', filepath=None, radar_type=42, protocal='RS485', bufflen=1000):
        self.port = port
        self.messages = {
            '0320': '0320_sensor_echo',
            '02ff': '02ff_sync_msg',
            '0734': '0734_coprocessor_status_msg',
            '0500': '0500_sensor_status', #contains timestamp
            '0501': '0501_object_data', #detected objects
            '0700': '0700_sensor_setup_msg', #xyz position/rotation
            '0780': '0780_info_msg',
            '0781': '0781_feature_msg',
            '0782': '0782_time_msg',
            '0783': '0783_relay control(5)/presence/wrong direction(2) msg', #distinguish by length
            '0784': '0784_triggered_objects_msg',
            '0785': '0785_PVR header(7)/PVR object message(2)', #distinguish by length
        }
        self.current_objects = [] #record newest object from dadar
        self.work_time = None #recode newest radar working time(ms) as timestamp
        self.filepath = filepath
        self.radar_type = radar_type
        self.protocal = protocal
        self.bufflen = bufflen
        self.buff = [{} for x in range(self.bufflen)] #[[] for x in range(bufflen)] #[{'worktime_ms': t, 'oid1': {}, 'oid2': {}, ...}]
        self.buff_idx = -1
        self.isstop = False
        self.start_systime = None   #record first radar data's system time
        self.start_worktime = None  #record first radar data's working time

    def get_info(self):
        for port, desc, hwid in sorted(serial.tools.list_ports.comports()):
            print("{}: {} [{}]".format(port, desc, hwid))

    def parse(self):
        self.ser = RadarOffline(self.filepath) if self.filepath else serial.Serial(self.port, baudrate=115200, #timeout=10)
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, 
            timeout=None, xonxoff=False, rtscts=False, write_timeout=None, 
            dsrdtr=False, inter_byte_timeout=None, exclusive=None
            )
        self.wrok_time = 0 #reset         
        state = 'init'
        header_len = payload_len = 0
        raw_data = ''
        while (not self.isstop):
            s = self.ser.read(1)
            if s == b'\x7e' and state == 'init':
                raw_data = '' #clear raw_data
                state = 'protocol_ver'
                if not self.start_systime:
                    self.start_systime = time()
            elif state == 'protocol_ver':
                state = 'header_len'
            elif state == 'header_len':
                header_len = int.from_bytes(s, byteorder='big')
                raw_data += s.hex()
                # process data
                #read remaining headers
                s = self.ser.read(header_len-3)
                raw_data += s.hex() + ' '
                payload_len = int.from_bytes(s[0:2], byteorder='big')
                s = self.ser.read(payload_len) #read payload
                payload = s.hex()
                raw_data += s.hex() + ' '
                s = self.ser.read(2) #read CRC
                raw_data += s.hex()
                Log.debug(raw_data) #print raw_data
                # parse payload
                try:
                    ret = self.parse_payload(payload)
                    Log.info(ret) # print decoded_data
                except:
                    Log.info('Payload decode error!')
                # reset state
                state = 'init'
                continue

            raw_data += s.hex()

    def parse_payload(self, payload):
        self.current_objects = {} #[] #reset
        ret = []
        ptr = 0 # indicate decode position
        while ptr < len(payload):
            # parse data columns
            ident = payload[ptr:ptr+4]
            if ident == '00': # special weird case
                # print('weird case', ident)
                return ret
            length = int(payload[ptr+4:ptr+6], 16)
            data = payload[ptr+6:ptr+6+length*2]
            ptr += length*2 + 6
            # decode data
            decoded_data = self.decode(ident, data)
            ret.append({self.messages[ident]: decoded_data} if self.messages.get(ident) else {ident: decoded_data})

        #update buffer
        if self.current_objects:
            self.buff_idx = (self.buff_idx + 1) % self.bufflen
            self.buff[self.buff_idx] = self.current_objects.copy()
            for i in range(self.bufflen):
                if not self.buff[i]:
                    self.buff[i] = self.current_objects.copy()

        return ret

    def get_object_type(self, length):
        '''according to [en_42] sec. 7.2.1, pp.70'''
        if length <= 1.0:
            return 'pedistrian'
        elif 1.0 < length <= 1.6:
            return 'bicycle'
        elif 1.6 < length <= 2.6:
            return 'motorcycle' #'bike/motorcycle'
        elif 4.6 <= length <= 5.4:
            return 'sedan/taxi/suv' #'passenger car'
        elif 5.6 <= length <= 8.8:
            return 'delivery/pickup'
        elif 9.0 <= length <= 13.8:
            return 'short truck'
        elif 14.0 <= length:
            return 'long truck'
        else:
            return 'undefined'

    def decode(self, ident, data):
        #rearrange hex data (2 by 2)
        data = ''.join([data[x]+data[x+1] for x in reversed(range(0, len(data), 2))])
        ret = {'data': data} #default output
        if ident == '0500': #'sensor status' -> get working time
            ret = {
                'worktime_ms': int(data[0:8], 16),
                'device_id': int(data[12], 16),
            }          
            self.work_time = ret['worktime_ms']
            self.current_objects.update({'worktime_ms': ret})
            # system time
            if not self.start_worktime:
                self.start_worktime = ret['worktime_ms']
            self.current_objects.update({'systime': self.start_systime + (ret['worktime_ms'] - self.start_worktime) / 1000})
            # print(self.work_time)

        elif ident == '0501': # Objects Control Message [ch_42] sec.4.3.2 pp.14
            ret = {
                'cycle_count': int(data[0:8], 16),
                'object_data_format_#1': int(data[8], 16),
                'object_data_format_#0': int(data[9], 16),
                'cycle_duration': int(data[10:12], 16),
                'num_message': int(data[12:14], 16),
                'num_target': int(data[14:16], 16),
            }

        elif ident.startswith('05'): # [en_42] sec.7.2.1 pp.35, [ch_42] sec.4.3.3 pp.17
            bins = bin(int(data, 16))[2:].zfill(len(data)*4) #hex to bits, ref: https://stackoverflow.com/a/4859937/10373104
            # bins = format(int(data, 16), '0>%db' % (len(data)*4)) #hex to bits, ref: https://stackoverflow.com/a/37221884/10373104
            ret = {
                'oid': int(bins[0:8], 2),
                'length': int(bins[8:15], 2) * 0.2, #meter
                'speed_y': (int(bins[15:26], 2) - 1024) * 0.1, #m/s
                'speed_x': (int(bins[26:37], 2) - 1024) * 0.1, #m/s
                'y': (int(bins[37:50], 2) - 4096) * 0.128, #meter
                'x': (int(bins[50:63], 2) - 4096) * 0.128, #meter
                'updated_flag': int(bins[63], 2),
                'raw_data': data.split('\n')[0],
            }            
            ret.update({'type': self.get_object_type(ret['length'])})
            self.current_objects.update({'worktime_ms': self.work_time})
            self.current_objects.update({ret['oid']: ret})

        elif ident == '02ff': #sync message -> indicate new data cycle -> ?
            ret = {'worktime_s':  int(data[14:16], 16)*0.008} #[ch_42] sec.4.2.2 pp.11

        elif ident == '0780': #info message -> get timestamp
            bins = bin(int(data, 16))[2:].zfill(len(data)*4) #hex to bits
            if len(bins) == 64:
                if bins[63] == 0: #info message #1
                    ret = {
                        'sensor_id': int(bins[8:12], 2),
                        'num_lane': int(bins[56:60], 2),
                    }
                else: #info message #2
                    ret = {
                        'timestamp_s': int(bins[0:32], 2),
                        'timestamp_ms': int(bins[53:63], 2),
                    }
            else:
                ret = {'raw_data': data}
            ret.update({'raw_data': data})

        return ret

    def start_parse(self):
        Thread(target=self.parse, args=(), daemon=True).start()

    def stop(self):
        self.isstop = True

    def get_current_object(self, time_ms, nearest=False):
        '''generate current object for given timestamp'''
        best_idx, best_diff, best_time = None, 9999999999, None

        for i in range(self.bufflen):
            if self.buff[i]:
                diff = abs(self.buff[i]['worktime_ms'] - time_ms)
                if diff < best_diff:
                    best_idx = i
                    best_diff = diff
                    best_time = self.buff[i]['worktime_ms']
        if best_idx is None: #buff are all empty
            return None

        ret = self.buff[best_idx] #deepcopy(self.buff[best_idx]) #default return
        
        if nearest:
            return ret

        prev_objs = self.buff[best_idx-1] #previous objs
        next_objs = self.buff[(best_idx+1)%self.bufflen] #next objs
        prev_t = prev_objs.get('worktime_ms', 9999999999)#[0]['worktime_ms'] #previous timestamp
        next_t = prev_objs.get('worktime_ms', -1)#[0]['worktime_ms'] #next timestamp

        def interpolation(key, t, x1, x2, t1, t2): #interpolation by position
            return x1[key] + (x2[key] - x1[key]) * (t - t1) / (t2 - t1) if t1 != t2 else x1[key]
        def interpolation1(key, t, x1, x2, t1, t2): #interpolatin by speed
            return x1[key] + (x1['speed_x'] if 'x' in key else x1['speed_y']) * (t - t1) / 1000

        #loop each object of best_idx
        for oid, val in self.buff[best_idx].items():
            if oid in ['worktime_ms', 'systime']:
                continue
            # -----prev-----c1-----best-----c2-----next-----
            if best_time > time_ms:
                if time_ms > prev_t and prev_objs.get(oid): #c1 case -> interpolation
                    # print('c1', end=' ')
                    x1, t1, x2, t2 = prev_objs[oid], prev_t, val, best_time
                elif next_objs.get(oid): #prev_t is larger or oid not found -> extrapolation
                    # print('c1a', end=' ')
                    x1, t1, x2, t2 = val, best_time, next_objs[oid], next_t
                else:
                    # print('c1b', end=' ')
                    x1, t1, x2, t2 = val, best_time, val, best_time
            else:
                if time_ms < next_t and next_objs.get(oid): #c2 case -> interpolation
                    # print('c2', end=' ')
                    x1, t1, x2, t2 = val, best_time, next_objs[oid], next_t
                elif prev_objs.get(oid): #next_t is smaller or oid not found -> extrapolation
                    # print('c2a', end=' ')
                    x1, t1, x2, t2 = prev_objs[oid], prev_t, val, best_time
                else:
                    # print('c2b', end=' ')
                    x1, t1, x2, t2 = val, best_time, val, best_time

            ret.update({oid: {
                'oid': oid, 'length': val['length'], 'type': val['type'], 'updated_flag': val['updated_flag'], 
                'raw_data': None, 'worktime_ms': time_ms,
                'speed_y': val['speed_y'], #interpolation('speed_y', time_ms, x1, x2, t1, t2),
                'speed_x': val['speed_x'], #interpolation('speed_x', time_ms, x1, x2, t1, t2),
                'y': interpolation1('y', time_ms, x1, x2, t1, t2),
                'x': interpolation1('x', time_ms, x1, x2, t1, t2),
            }})

        ret.update({'worktime_ms': time_ms})
        return ret


class IPcamCapture:
    '''
    # 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
    ref: https://dotblogs.com.tw/shaynling/2017/12/28/091936
    '''
    def __init__(self, URL, offline=False, start_frame=0):
        self.offline = offline
        self.buff = deque(maxlen=100)
        self.is_stop = False
        self.drift_time = timedelta(0)
        self.start_time = datetime.strptime(URL.split('\\')[-1].split('.')[0], "%Y-%m-%d_%H-%M-%S,%f") if offline else time()

        # 攝影機連接。
        self.cap = cv2.VideoCapture(URL)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) #float
        self.w_h = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(self.w_h)

    def start(self, write=False):
        '''開始串流        
        把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        '''
        print('IPcam started!', flush=True)
        if not self.offline:
            Thread(target=self.query_frames, daemon=True, args=()).start()
        if write:
            sleep(0.1) #prevent write before reading
            Thread(target=self.write_file, daemon=True, args=()).start()

    def stop(self):
        '''停止串流'''
        self.is_stop = True
        print('IPcam stopped!')
        sleep(1) #確保ipcam thread結束

    def query_frames(self):
        '''串流子序列
        只負責從攝影機讀取影像到緩衝器的副程式
        '''
        while(not self.is_stop):
            self.status, self.Frame = self.cap.read()
            self.buff.append({'frame': self.Frame, 'time': time()})
        self.cap.release()

    def get_frame(self):
        '''取得影像以及對應時間(經過調整)
        當有需要影像時，再回傳最新的影像。
        '''
        if not self.offline:
            current_time = self.buff[-1]['time'] + self.drift_time.total_seconds()
            current_frame = self.buff[-1]['frame'].copy()
            return current_frame, current_time*1000 
        else:
            self.status, current_frame = self.cap.read()
            cruurent_time = self.cap.get(cv2.CAP_PROP_POS_FRAMES)/self.fps + self.start_time.timestamp() + self.drift_time.total_seconds()
            return current_frame, cruurent_time*1000
        
    def get_time_bar(self, number_frames=30):
        '''取得n張frame的時間，用來檢查時間差'''
        timebars = []
        for ii in range(number_frames):
            current_frame = self.buff[-30+ii]['frame'].copy()
            current_time = self.buff[-30+ii]['time']
            time_string = datetime.fromtimestamp(current_time) + self.drift_time
            cv2.putText(current_frame, '%s'%(time_string.strftime("%Y-%m-%d  %H : %M : %S.%f")) , (10,60), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0,0,255),2,cv2.LINE_AA)
            timebars.append(current_frame[0:65,0:650])
        time_bar_image = np.vstack(timebars)
        time_bar_image = cv2.resize(time_bar_image, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('sys time', time_bar_image)
        
    def modify_time_drify(self, minute=0, second=0, microsecond=0):
        '''微調影像的落後系統時間
        若輸出的系統時間比影像內容快兩秒，就在在second的地方輸入+2
        '''
        self.drift_time = self.drift_time + timedelta(minutes=minute, seconds=second, microseconds=microsecond)

    def write_file(self):
        '''寫入影像至檔案'''
        while len(self.buff) == 0:
            pass
        filename = datetime.fromtimestamp(self.buff[-1]['time']).strftime("%Y-%m-%d_%H-%M-%S,%f")
        vout = cv2.VideoWriter("logs/%s.mp4" % filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.w_h, True)
        print('write to %s' % filename)
        f = 0 #current frame
        start_time = time()
        while (not self.is_stop):
            t1 = f / self.fps + start_time
            for i in range(len(self.buff)): #find nearest frame
                if self.buff[i-1]['time'] <= t1 <= self.buff[i]['time']: #found a frame to write
                    idx = i if (self.buff[i]['time'] - t1) < (t1 - self.buff[i-1]['time']) else i-1
                    video_img = self.buff[idx]['frame'].copy()
                    cv2.putText(video_img, str(datetime.now()), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255), 2, cv2.LINE_AA)
                    vout.write(video_img)
                    f += 1
                    # sleep(1/self.fps/2)
                    break
        vout.release()
        print('realease')


class RadarOfflineOld:
    def __init__(self, filepath):
        self.f = open(filepath, 'r')
        self.buff = '' #buffering raw data (hex)
        self.ptr = 0 #pointer of raw data reading position

        if not self.f:
            print('file not found', flush=True)
            sys.exit()

        print('reading radar offline file... ', end='\n', flush=True)
        while True:
            line = self.f.readline()
            if not line:
                return
            #Allen ver.
            if line.startswith('['): #debug info -> pass
                continue
            #III ver.
            keyword = line.split()[3]
            # print('.', end='', flush=True)
            if keyword == 'INFO': #not raw data
                continue
            elif keyword == 'DEBUG':
                line = line.split('DEBUG')[1]

            self.buff += line.replace('\n', '').replace(' ', '')
        print('done', flush=True)

    def read(self, num_byte):
        '''simulate serial byte reading'''
        # sleep(0.018)
        sleep(0.00001)
        self.ptr += num_byte*2
        return bytes.fromhex(self.buff[self.ptr-num_byte*2 : self.ptr])


class RadarOffline:
    def __init__(self, filepath):
        self.buff = [] #buffering raw data (hex)
        self.ptr = 0 #pointer of raw data reading position
        # self.t0 = datetime.now() #for simulating radar timestamp
        
        #open and read radar log file
        f = open(filepath, 'r')
        if not f:
            print('file not found', flush=True)
            sys.exit()

        print('reading radar offline file... ', end='', flush=True)
        self.log_t0 = datetime.strptime(' '.join(f.readline().split()[0:2]), '%Y-%m-%d %H:%M:%S,%f')
        f.seek(0)
        while True:
            line = f.readline()
            if not line:
                print('done', flush=True)
                self.t0 = datetime.now()
                return
            (date, time, _, keyword) = line.split()[0:4]
            if keyword == 'INFO': #not raw data
                continue
            elif keyword == 'DEBUG':
                data = line.split('DEBUG')[1].replace('\n', '').replace(' ', '')

            self.buff.append({'data': data, 'time': datetime.strptime('%s %s' % (date, time), '%Y-%m-%d %H:%M:%S,%f')})

    def read(self, num_byte):
        '''simulate serial byte reading'''
        while True:
            if (datetime.now() - self.t0) >= (self.buff[0]['time'] - self.log_t0): #current time over radar data input time
                # print(datetime.now(), self.t0, self.buff[0]['time'], self.log_t0)
                ret, self.buff[0]['data'] = self.buff[0]['data'][:num_byte*2], self.buff[0]['data'][num_byte*2:] #pop desired bytes
                if len(self.buff[0]['data']) == 0: #switch to next data if current is empty
                    self.buff.pop(0)
                # print('%.1f %.1f' % (datetime.now().timestamp(), self.buff[0]['time'].timestamp()))
                # print('%.1f' % (self.buff[0]['time'].timestamp() - self.log_t0.timestamp() + 9119380), end=' ')
                return bytes.fromhex(ret)
            # sleep(0.005)


class RadarData(object):
    '''讀取Offline 雷達儲存資料
    
    從Terry儲存的雷達Log中找到雷達資訊，並將每個時間點的數值以時間戳記儲存在一個list內。
    
    '''
    def __init__(self, filepath):
        self.buff_raw_data = '' #buffering raw data (hex)
        self.ptr = 0 #pointer of raw data reading position
        self.objects  = []
        self.list_time_total_second = []
        self.list_object_frame = []
        self.headtime = None
        self.headtimems =None
        self.read_file(filepath)
        
    def read_file(self, filepath):
        self.f = open(filepath, 'r')
        find_headtime = False
        if not self.f:
            print('file not found', flush=True)
            sys.exit()
            
        print('reading radar offline file... ', end='\n', flush=True)
        while True:
            ## read line and check avalueable
            line = self.f.readline()
            
            if not line:
                break
            if line.startswith('['):
                continue

            keyword = line.split()[3]
            if keyword == 'INFO': #not raw data
                ## Initial head time
                if not find_headtime:
                    self.headtime  = datetime.strptime(line[:19],"%Y-%m-%d %H:%M:%S")
                    self.headtimems = int(line.split('worktime_ms')[1].split(',')[0].split()[1])
                    find_headtime = True     
                    
                ## Make INFO format match json format
                list_json_string = line.split('INFO')[1].replace('\'','\"').split('{\"05')
                if len(list_json_string)==1:
                    continue
                list_json_string = list_json_string[1:]
                list_json_string[-1] = list_json_string[-1].split('{\"0780')[0]
                
                ## Read sensor time and objects
                dict_current_objects= {}
                for json_string in list_json_string:
                    dict_object = json.loads('{\"'+json_string[:-2])
                    if list(dict_object.keys())[0]=='00_sensor_status':
                        ## Sensor time drift info
                        drift_time_ms = dict_object['00_sensor_status']['worktime_ms']-self.headtimems
                        current_time = self.headtime + timedelta(microseconds  = drift_time_ms*1000 )
                        sample_time= current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                        dict_current_objects.update({'time':sample_time })

                    elif list(dict_object.keys())[0]!='01_object_data':
                        ## Object info
                        dict_current_objects.update(dict_object)

                ## update current time status
                self.list_time_total_second.append( (current_time-datetime(1970,1,1)).total_seconds() )
                self.list_object_frame.append(dict_current_objects)
                
            elif keyword == 'DEBUG': ## raw data
                line = line.split('DEBUG')[1]
                self.buff_raw_data += line.replace('\n', '').replace(' ', '')
        print('done', flush=True)

    def read(self, timestamep):
        '''讀取指定時間點的雷達資料
        
        以時間戳記字串("%Y-%m-%d %H:%M:%S.%f")來讀取目前讀取時間內的Offline雷達資料。
        輸出為最接近該時間點的雷達資料點位。

        Parameters
        ----------
        timestamep : str
            需求的時間戳記字串，格式為"%Y-%m-%d %H:%M:%S.%f"。

        Returns
        -------
        TYPE dict
            雷達點位資料，以字典的方式記錄各個點位，各點位包含Oid, x, y, speed x, speedy等等資訊

        '''
        
        current_time = datetime.strptime(timestamep,"%Y-%m-%d %H:%M:%S.%f")
        total_seconds = (current_time-datetime(1970,1,1)).total_seconds()
        nearest_time = np.argmin(np.abs(np.array(self.list_time_total_second)-total_seconds))

        return self.list_object_frame[nearest_time].copy()


#%%%

def main(radar):
        #init arguments
        radar_pos_latlong = (25.058857, 121.553933) #(25.058857, 121.553950)#(25.058857, 121.553933) #default
        radar_angle = 26#50 #26 #degree originated from North
        ratio = 1.0 #radar image resize ratio
        #mingshen2a, 3
        shift_x, shift_y, radius = 300, -350, 400
        #mingshen4
        shift_x, shift_y, radius = 750, -330, 400

        #read real (radar) image
        orig_img = cv2.imread(r"test\mingshen4.jpg")
        resized_img = cv2.resize(orig_img, (int(orig_img.shape[1]*ratio), int(orig_img.shape[0]*ratio)))
        center_x, center_y = int(resized_img.shape[0]/2), int(resized_img.shape[1]/2)
        radar_img = resized_img[center_y+shift_y-radius:center_y+shift_y+radius, center_x+shift_x-radius:center_x+shift_x+radius]

        #read video image        
        # print('load video', flush=True)
        start_frame = 0#50
        dataloader = LoadVideo(start_frame=start_frame,
            # url = 'rtsp://192.168.0.100/media.amp?streamprofile=Profile1', #from rtsp
            base_path='test', video_name='2020-03-30_11-51-39', #from file
        )
        fps, width, height, field_path, src = dataloader.get_info()  
        dataloader.__iter__()
        video_img, orig_img, curr_frame, timer = dataloader.__next__()
        
        #video output
        h = 500   
        vout = cv2.VideoWriter("test/0412_radar.avi", cv2.VideoWriter_fourcc(*'XVID'), fps/2, (radius*4, radius*2+h), True)
        TMC_video = LoadVideo(start_frame=-870, base_path='test', video_name='雷達_2020_03_30_11_52_13')
        TMC_video.__iter__()
        TMC_img, TMC_orig_img, TMC_curr_frame, TMC_timer = TMC_video.__next__()
        perspective_TMC = [(1235, 380), (1253, 330), (1323, 453), (1275, 450)]
        cv2.polylines(TMC_img, [np.array(perspective_TMC, np.int32)], True, (0, 0, 255), 2)

        #perspective transform
        #mingshen2a
        perspective_radar = [(361-15, 380), (356, 294), (490, 533), (448-15, 538)]
        #mingshen3
        perspective_radar = [(405, 540), (425, 485), (518, 650), (470, 658)]
        #mingshen4
        perspective_radar = [(402, 540), (422, 485), (515, 650), (467, 658)]
        latlong_radar = [(25.059207, 121.553626), (25.059319, 121.553628), (25.059000, 121.553820), (25.058994, 121.553752)]
        perspective_video  = [(790, 160), (1040, 160), (930, 430), (290, 380)]
        PT = PerspectiveTransform(perspective_radar, perspective_video, latlong_radar)

        #draw radar position
        pos_radar = np.add(PT.latlong_to_pixel(np.subtract(radar_pos_latlong, latlong_radar[0])), perspective_radar[0]).astype(int)
        cv2.circle(radar_img, tuple(pos_radar), 3, (40, 255, 175), -1, cv2.LINE_8)
        cv2.putText(radar_img, 'radar', tuple(pos_radar), cv2.FONT_HERSHEY_PLAIN, 1.5, (140, 220, 90), 2, cv2.LINE_AA)           

        #draw radar anchor (pointing direction)
        rad = (270-radar_angle) / 180 * np.pi
        radar_anchor = np.add(np.multiply((np.cos(rad), np.sin(rad)), 500), pos_radar).astype(int)
        cv2.arrowedLine(radar_img, tuple(pos_radar), tuple(radar_anchor), (153, 255, 255), 2, tipLength=0.01)

        #draw perspective regions
        cv2.polylines(radar_img, [np.array(perspective_radar, np.int32)], True, (0, 0, 255), 2)
        cv2.polylines(video_img, [np.array(perspective_video, np.int32)], True, (0, 0, 255), 2)

        #debug perspective
        # cv2.imshow('TMC', TMC_img[:,1000:])
        # cv2.imshow('video', video_img)
        # cv2.imshow('radar', radar_img)
        # cv2.waitKey(0), sys.exit()

        radar.start_parse()
        radar_start_time = None
        while True:
            if not radar_start_time: #do until get radar start time
                radar_start_time = radar.work_time
                continue
            video_img, _orig_img, curr_frame, timer = dataloader.__next__()
            TMC_img, TMC_orig_img, TMC_curr_frame, TMC_timer = TMC_video.__next__()
            print(curr_frame-start_frame, end=' ')

            #draw perspective region in video image
            cv2.polylines(video_img, [np.array(perspective_video, np.int32)], True, (0, 0, 255), 2)
            #draw perspective region in TMC image
            cv2.polylines(TMC_img, [np.array(perspective_TMC, np.int32)], True, (0, 0, 255), 2)

            current_objects = radar.get_current_object(timer * 1000 + radar_start_time, nearest=False)
            if current_objects:
                print('%.3fs' % timer, '%.3fs' % ((current_objects['worktime_ms']-radar_start_time)/1000), end=' ', flush=True)
                for key, val in current_objects.items():
                    if key != 'worktime_ms':
                        print(' #%d(%.2f,%.2f)' % (val['oid'], val['y'], val['x']), end='')
            else:
                print('%.3fs' % timer, '*', end='', flush=True)

            #start drawing
            img = radar_img.copy()
            for oid, _obj in current_objects.items():
                if oid == 'worktime_ms':
                    continue
                # rotation & flipping
                flip = 1 #1(no) or -1(flipping)
                obj = deepcopy(_obj)#_obj.copy() #make a copy since radar.current_objects is keep updating
                norm, rad = np.linalg.norm((flip*obj['y'], obj['x'])), np.arctan2(obj['x'], flip*obj['y']) + radar_angle/180*np.pi
                norm_sp, rad_sp = np.linalg.norm((flip*obj['speed_y'], obj['speed_x'])), np.arctan2(obj['speed_x'], flip*obj['speed_y']) + radar_angle/180*np.pi
                obj.update({
                    'y': norm*np.cos(rad), 
                    'x': norm*np.sin(rad),
                    'speed_y': norm_sp*np.cos(rad_sp), 
                    'speed_x': norm_sp*np.sin(rad_sp),
                })
                pos = tuple(np.add(PT.meter_to_pixel((obj['y'], obj['x'])), pos_radar).astype(int))
                pos1 = tuple(np.add(PT.meter_to_pixel((obj['speed_y'], obj['speed_x'])), pos).astype(int))

                # draw radar image
                show_text = '%d %s' % (obj['oid'], obj['type'])
                cv2.circle(img, pos, 3, (40, 255, 175), -1, cv2.LINE_8)
                cv2.putText(img, show_text, pos, cv2.FONT_HERSHEY_PLAIN, 1.5, (140, 220, 90), 2, cv2.LINE_AA)
                cv2.arrowedLine(img, pos, pos1, (255, 255, 0), 2, tipLength=0.2)

                #perform perspective projection https://en.wikipedia.org/wiki/Transformation_matrix#Perspective_projection
                p = PT.project_to_pixel(pos)
                p1 = PT.project_to_pixel(pos1)

                # draw video image
                cv2.circle(video_img, p, 3, (40, 255, 175), -1, cv2.LINE_8)
                cv2.putText(video_img, show_text, p, cv2.FONT_HERSHEY_PLAIN, 1.5, (140, 220, 90), 2, cv2.LINE_AA)
                cv2.arrowedLine(video_img, p, p1, (255, 255, 0), 2, tipLength=0.2)

            # cv2.imshow('real', video_img)            
            cv2.imshow('radar', img)
            # cv2.imshow('', TMC_img[0:radius*2, (TMC_img.shape[1]-radius*2):-1])
            print('')

            #video output
            if vout:
                TMC_output = cv2.resize(TMC_img[0:radius*2, (TMC_img.shape[1]-radius*2):-1], (radius*2, radius*2))
                out = np.concatenate((np.concatenate((img, TMC_output), axis=1), video_img[0:h, 0:radius*4]), axis=0)
                # out = np.concatenate((img, TMC_output), axis=1)
                # cv2.imshow('ee', video_img[0:h, 0:radius*4])#cv2.resize(out, (800, 650)))                
                vout.write(out)

            if cv2.waitKey(1) & 0xFF in [ord('q'), ord("Q"), 27]:
                # cv2.destroyAllWindows()
                if vout:
                    vout.release()
                sys.exit()
            sleep(1/fps) 

def main_exp(radar, vout):
    # 連接攝影機
    ipcam = IPcamCapture("rtsp://keelung:keelung@nmsi@60.251.176.43/onvif-media/media.amp?streamprofile=Profile1")
    fps = FPSCounter()
    
    # 啟動子執行緒
    ipcam.start(write=vout)
    # 暫停1秒，確保影像已經填充
    sleep(0.5)

    # 啟動雷達logging
    if radar:
        radar.start_parse()

    # 使用無窮迴圈擷取影像，直到按下Esc鍵結束
    while True:
        fps.tic()
        # 使用 getframe 取得最新的影像
        img, _ = ipcam.getframe()
                
        t = fps.toc()
        fps_curr = fps.update_fps(t)
        cv2.putText(img, "FPS: %d" % fps_curr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow('Image', img)

        if cv2.waitKey(1) in [ord('q'), ord("Q"), 27]:
            cv2.destroyAllWindows()
            ipcam.stop()
            break        

def set_RWV_transformer(src, radar=None):
    #初始雷達對齊    
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
    if os.path.exists('points_image.npz'):
        points = np.load('points_image.npz')
        image_points, world_points = points['arr_0'], points['arr_1']
    else:
        _, img = cv2.VideoCapture(src).read()
        points = MousePointsReactor(img, len(world_points), ['x', 'y'], world_points).start()
        world_points = [world_points[x] for x in points.keys()]
        image_points = list(points.values())
        np.savez('points_image.npz', image_points, world_points)
    
    
    rt = RSVTransformer()
    #set1
    rt.set_image_points(image_points)
    rt.set_world_points(world_points)
    rt.load_camera_parameter_by_path(os.path.join(os.getcwd(), 'event'))#r'C:\\GitHub\\109_RadarFusion\\panasonic_camera\\')
    rt.set_new_camera_matrix(np.array([[1805.41,	   0,	924.743],
                                       [      0,	1100,	539.679],
                                       [      0,	   0,	      1]]))
    rt.calculate_world_to_image_matrix()

    #set2
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

    # mouse configuring or load existance
    check_len = len(world_points)
    if os.path.exists('points_radar.npz'):
        world_points = np.load('points_radar.npz')['arr_0']
        radar_points = np.load('points_radar.npz')['arr_1']
    else:
        _, img = cv2.VideoCapture(src).read()
        points = TKInputDialog(
            labels=['pt%d' % x for x in range(check_len)], 
            defaults=[str(world_points[x].tolist()).strip('[]')+', '+str(radar_points[x].tolist()).strip('[]') for x in range(check_len)], 
            title='give real world & radar points\n (x, y, a, b)', 
            win_size=(200, 50 * check_len)
        ).get()
        points = [[float(y) for y in x.split(',')] for x in points]
        world_points, radar_points = [x[0:2] for x in points], [x[2:4] for x in points]
        np.savez('points_radar.npz', world_points, radar_points)

    rt.set_radar_points(radar_points)
    rt.set_world_points(world_points)
    rt.calculate_radar_world_matrix()
    return rt

def draw_birdview_basemap(scale, wh):
    radar_img = np.zeros((wh, wh, 3), np.uint8)
    for i in range(3): #draw lane line
        x = int(wh/2 - i*6*scale)
        cv2.line(radar_img, (x, 0), (x, wh), (255, 255, 255), 1)
    for i in range(10): #draw distance line
        x, y = int(wh/2 - 6*scale), int(i*10*scale)
        cv2.line(radar_img, (x-5, y), (x+5, y), (255, 255, 255), 1)
        cv2.putText(radar_img, str(int(100-i*10)), (0, y+5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return radar_img

def main_video_radar(src, COM='COM13', filepath=None):

    #是否為離線模式    
    offline = not src.startswith('rtsp')
    filepath = None if src.startswith('rtsp') else filepath
    vout = src.startswith('rtsp')

    ## 設置投影矩陣
    rt = RSVTransformer()
    rt.load_parametre(path_load='RSV_metrix.npz')
    
    ## 連接攝影機
    ipcam = IPcamCapture(src, offline=offline, start_frame=0)
    ipcam.modify_time_drify(second=2, microsecond=200000)
    fps = FPSCounter()
    # 啟動子執行緒
    ipcam.start(write=vout)
    # 暫停1秒，確保影像已經填充
    sleep(0.5)

    ## 啟動雷達logging
    radar = RadarCapture(COM, filepath=filepath)
    radar.get_info()
    radar.start_parse()
    
    # 使用無窮迴圈擷取影像，直到按下Esc鍵結束
    radar_start_time = system_start_time = None
    static_points = {} #紀錄雷達定點
    scale, radar_img_wh = 6, 600 #雷達點畫圖放大倍率、雷達底圖長寬
    radar_base_img = draw_birdview_basemap(scale, radar_img_wh) #雷達鳥瞰底圖
    sizes = { #雷達物件點大小
        'pedistrian': 1, 'bicycle': 3, 'undefined': 3,
        'motorcycle': 4, 'bike/motorcycle': 4,
        'sedan/taxi/suv': 6, 'passenger car': 6,
        'delivery/pickup': 9, 'short truck': 9, 'long truck': 9,
    }
    while True:
        fps.tic()
        ## 設立迴圈開始時間，統一以milisecond紀錄。
        if not radar_start_time: #do until get radar start time
            system_start_time = time()*1000 if not offline else datetime.strptime(src.split('\\')[-1].split('.')[0], "%Y-%m-%d_%H-%M-%S,%f").timestamp()*1000
            radar_start_time = radar.work_time
            continue

        # 使用 getframe 取得最新的影像(時間為攝影機調整過時間)
        video_img, last_video_time = ipcam.get_frame()
                
        # 以影片時間找到對應的雷達時間
        radar_drift_time = timedelta(milliseconds=1).total_seconds()
        search_radar_time = radar_start_time + (last_video_time - system_start_time) + radar_drift_time
        current_radar_objects = radar.get_current_object(search_radar_time, nearest=True)
        
        radar_img = radar_base_img.copy()
        (size, color, color2, color3, color4) = (3, (40, 255, 175), (140, 220, 90), (255, 255, 0), (0, 204, 204))
        if current_radar_objects:
            ## Transformer Radar to Video Image
            radar_objects = current_radar_objects.copy()
            radar_objects.pop('worktime_ms')
            radar_objects.pop('systime')
            oids = list(radar_objects.keys())
            radar2image_points = rt.transepose_radar_to_image([(obj['x'], obj['y']) for oid, obj in radar_objects.items()])

            ##start drawing
            # Video
            for i, p in enumerate(radar2image_points):
                p0 = tuple(p[::-1].astype(int).tolist())
                show_text = '%d %s' % (oids[i], radar_objects[oids[i]]['type'])
                cv2.circle(video_img, p0, sizes[radar_objects[oids[i]]['type']], color, -1, cv2.LINE_8)
                cv2.putText(video_img, show_text, p0, cv2.FONT_HERSHEY_PLAIN, 1, (140, 220, 90), 1, cv2.LINE_AA)
            
            # Birdview
            tmp_list = list(static_points.keys())
            for oid, obj in radar_objects.items():
                # print(obj)
                p0 = (int(-obj['y']*scale + radar_img_wh/2), int(radar_img_wh - obj['x']*scale))
                cv2.circle(radar_img, p0, sizes[obj['type']], color, -1, cv2.LINE_8)
                cv2.putText(radar_img, str(oid), (p0[0]+5, p0[1]), cv2.FONT_HERSHEY_PLAIN, 1, color3, 1, cv2.LINE_AA) #show only oid
                #set by moving away or towards
                # show_text = '%-3d %15s' % (oid, obj['type']) if obj['speed_x'] > 0 else '%-15s %3d' % (obj['type'], oid)
                # pt = (p0[0]+5, p0[1]) if obj['speed_x'] > 0 else (p0[0]-175, p0[1])
                # cv2.putText(radar_img, show_text, pt, cv2.FONT_HERSHEY_PLAIN, 1, color2, 1, cv2.LINE_AA)

                #considering speed
                if abs(obj['speed_x'])/3.6 < 0.2: #speed_x < xx km/h
                    if static_points.get(oid): #exists
                        tmp_list.remove(oid)
                        static_points[oid].append((obj['x'], obj['y']))
                        cv2.circle(radar_img, p0, sizes[obj['type']], (0, 0, 255), -1, cv2.LINE_8)
                        show_text = '(%.2f, %.2f)' % tuple(np.average(static_points[oid], axis=0))
                        cv2.putText(radar_img, show_text, (p0[0], p0[1]-12), cv2.FONT_HERSHEY_PLAIN, 1, color2, 1, cv2.LINE_AA)
                        #set by moving away or towards
                        # pt = (p0[0], p0[1]-12) if obj['speed_x'] > 0 else (p0[0]-170, p0[1]-12)
                        # cv2.putText(radar_img, '%-20s' % show_text if obj['speed_x'] > 0 else '%20s' % show_text, pt, cv2.FONT_HERSHEY_PLAIN, 1, color2, 1, cv2.LINE_AA)
                    else:
                        static_points[oid] = [(obj['x'], obj['y'])]
            for k in tmp_list: #remove non-static or abscent
                print(k, np.average(static_points[k], axis=0))
                static_points.pop(k)
        '''
        if current_radar_objects: #old ver
            ## Transformer Radar to Video Image
            # print('query:%.1f' % search_time, end=' ')
            # print(current_objects['worktime_ms'])
            radar_points =  [(obj['x'], obj['y']) for oid, obj in current_radar_objects.items() if oid not in ['worktime_ms', 'systime']]
            radar_speeds =  [(obj['speed_x'], obj['speed_y']) for oid, obj in current_radar_objects.items() if oid not in ['worktime_ms', 'systime']]
            radar_ids =     [obj['oid'] for oid, obj in current_radar_objects.items() if oid not in ['worktime_ms', 'systime']]
            radar_types =   [obj['type'] for oid, obj in current_radar_objects.items() if oid not in ['worktime_ms', 'systime']]
            radar2image_points = rt.transepose_radar_to_image(radar_points)

            ##start drawing
            # Video
            for i, p in enumerate(radar2image_points):
                p0 = tuple(p[::-1].astype(int).tolist())
                show_text = '%d %s' % (radar_ids[i], radar_types[i])
                (size, color) = (3, (40, 255, 175))
                cv2.circle(video_img, p0, size, color, -1, cv2.LINE_8)
                cv2.putText(video_img, show_text, p0, cv2.FONT_HERSHEY_PLAIN, 1, (140, 220, 90), 1, cv2.LINE_AA)
            
            # Birdview
            tmp_list = list(static_points.keys())
            (size, color, color2) = (3, (40, 255, 175), (140, 220, 90))
            for i, p in enumerate(radar_points):
                p0 = (int(-p[1]*scale + radar_img_wh/2), int(radar_img_wh - p[0]*scale))
                cv2.circle(radar_img, p0, size, color, -1, cv2.LINE_8)
                if radar_speeds[i][0] > 0: #move away
                    # show_text = ('%s %d' % (radar_types[i], radar_ids[i])).rjust(20)
                    show_text = '%-3d %15s' % (radar_ids[i], radar_types[i])
                    cv2.putText(radar_img, show_text, p0, cv2.FONT_HERSHEY_PLAIN, 1, color2, 1, cv2.LINE_AA)
                else: #towards
                    show_text = ('%-15s %3d' % (radar_types[i], radar_ids[i])).rjust(20)
                    cv2.putText(radar_img, show_text, (p0[0]-180, p0[1]), cv2.FONT_HERSHEY_PLAIN, 1, color2, 1, cv2.LINE_AA)

                #considering speed
                if abs(radar_speeds[i][0]) < 1*3.6: #speed_x < xx km/h
                    if static_points.get(radar_ids[i]): #exists
                        tmp_list.remove(radar_ids[i])
                        static_points[radar_ids[i]].append(p)
                        cv2.circle(radar_img, p0, size, (0, 0, 255), -1, cv2.LINE_8)
                        avg = np.average(static_points[radar_ids[i]], axis=0)
                        show_text = '(%.2f, %.2f)' % (avg[0], avg[1])
                        if radar_speeds[i][0] > 0: #move away
                            cv2.putText(radar_img, '%20s' % show_text, (p0[0], p0[1]-12), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
                        else: #towards
                            cv2.putText(radar_img, '%-20s' % show_text, (p0[0] - 170, p0[1]-12), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
                    else:
                        static_points[radar_ids[i]] = [p]
            for k in tmp_list: #remove non-static or abscent
                print(k, np.average(static_points[k], axis=0))
                static_points.pop(k)
        '''
        t = fps.toc()
        cv2.imshow('radar', radar_img)
        cv2.putText(video_img, "FPS: %d" % fps.update_fps(t), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(video_img, str(datetime.now()), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(video_img, str(datetime.fromtimestamp(last_video_time/1000)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow('video', video_img)
        if cv2.waitKey(1) in [ord('q'), ord("Q"), 27]:
            cv2.destroyAllWindows()
            ipcam.stop()
            break

#%%    
if __name__ == '__main__':
    from RSV_Transformer import RSVTransformer, PerspectiveTransform
    from RSV_ScopeSetting import MousePointsReactor, TKInputDialog

    dict_carmer = {
        'KL' : "rtsp://keelung:keelung@nmsi@60.251.176.43/onvif-media/media.amp?streamprofile=Profile1",
        'KH' : "rtsp://demo:demoIII@221.120.43.26:554/stream0",
        'IP' : 'rtsp://192.168.0.100/media.amp?streamprofile=Profile1',
        'LL' : r'F:\GDrive\III_backup\[Work]202006\0604雷達資料\0604cc\2020-06-04_17-03-11,698371.mp4'
    }
    filepath = r'F:\GDrive\III_backup\[Work]202006\0604雷達資料\0604cc\radar.log'
    main_video_radar(dict_carmer['KL'], COM='COM13', filepath=filepath)
    sys.exit()

    ''' test mode'''
    # radar.parse()
    # main(radar)
    # main_exp1(vout=True)
    # sys.exit()