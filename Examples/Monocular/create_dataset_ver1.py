# coding:utf-8

import os
import cv2
import numpy as np
from multiprocessing import Pool

class Kitti():
    def __init__(self, video_parent_dir, frame_interval, img_width, img_height):
        '''
        :param video_father_dir:  video father dir file dir
        :param frame_interval:   every frame_interval save a key frame
        :param img_width:   img sequence width
        :param img_height:  img sequence height
        '''

        self.video_dirs = [video_parent_dir+dir for dir in os.listdir(video_parent_dir)]
        self.frame_interval = frame_interval
        self.img_width = img_width
        self.img_height = img_height
        self.dir_cnt = 0

    def get_format_name(self, idx, lenght):
        '''
        :param idx: given img index, such as 1, 2, 3
        :lenght: format name length
        :return: return format img name like 000001, 000002, ...
        '''
        cnt = lenght - 1
        prefix = ''
        nmb = idx
        while idx // 10 != 0:
            cnt -= 1
            idx = idx // 10
        for i in range(cnt):
            prefix += '0'
        return prefix + str(nmb)
    def run(self,video_dir):
        videoCapture = cv2.VideoCapture(video_dir)
        # get video fps
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        # get vide width and height
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        width_offset = int((size[0] - self.img_width) / 2)
        height_offset = int((size[1] - self.img_height) / 2)
        print('=======================')
        print('init_size: ',size)
        print('dest_size: (',self.img_width,',',self.img_height,')')
        img_sequence_parentdir = './' + self.get_format_name(self.dir_cnt, 2)
        if not os.path.exists(img_sequence_parentdir):
            os.makedirs(img_sequence_parentdir)
            if not os.path.exists(img_sequence_parentdir + '/image_0/'):
                os.makedirs(img_sequence_parentdir + '/image_0/')
        
        success, frame = videoCapture.read()
        total_frame_idx = 0  # video frame index
        count = 0  # keyframe number
        tmp_cnt = 0  # record new cycle frame number
        timestep = 0  # time step
        timestep_total = [0]  # save every time step
        print("Fps: ", fps)
        print('=======================')

        idx=0
        while success:
        
            if tmp_cnt == 0 or tmp_cnt == self.frame_interval:
                format_img_name = img_sequence_parentdir + '/image_0/' +'frame'+str(idx)+'.png'
                print(format_img_name)
                cv2.imwrite(format_img_name,
                            frame[height_offset:height_offset + self.img_height,
                            width_offset:width_offset + self.img_width])
                count += 1
                tmp_cnt = 0
                timestep += self.frame_interval / fps
                timestep_total.append(timestep)

            success, frame = videoCapture.read()  # 获取下一帧
            total_frame_idx = total_frame_idx + 1

            tmp_cnt += 1
            idx+=1
        self.dir_cnt += 1
        np.savetxt(img_sequence_parentdir + '/times.txt', timestep_total)

    def main(self):
        # pool = Pool()
        # pool.map(self.run,self.video_dirs)
        # pool.close()
        # pool.join()

        for video_dir in self.video_dirs:
            self.run(video_dir)


if __name__ == "__main__":
    K = Kitti('../dataset/calib/',1,1024,682)
    K.main()
