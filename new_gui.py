import concurrent.futures
import csv
import datetime
# import the necessary packages
import os
import sqlite3
import threading
import time
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from tkinter import BOTH, Tk
from tkinter.ttk import Button, Frame, Label, Style
import cv2
import imutils
import numpy as np
import PIL
# import tensorflow as tf
import xlsxwriter
from imutils import contours, perspective
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
from ttkthemes import ThemedTk
from cdetect import V5
import shutil

# /home/kraken2/thread_GUIcopy/yolov4tflite/checkpoints/loom-tiny-640


def load_model():
    global PATH_SAVED_MODEL, yoloTiny, IMAGE_SIZE
    # PATH to saved and exported tensorflow model
    PATH_SAVED_MODEL = os.path.join(os.getcwd(), 'bestm.pt')
    IMAGE_SIZE = 640
    yoloTiny = V5( PATH_SAVED_MODEL)


class Example(Frame):

    def __init__(self, width, height, canvas):
        super().__init__()
        self.height = height
        self.width = width
        self.initUI()

    def submit(self, height1, width1, torn1, not_clean1, area1, result1, height2, width2, torn2, not_clean2, area2, result2, height3, width3, torn3, not_clean3, area3, result3, height4, width4, torn4, not_clean4, area4, result4):
        dbstart = time.perf_counter()
        conn = sqlite3.connect(
            'C:/Users/wahab/Desktop/thread/thread.db')
        c = conn.cursor()

        a = c.execute("SELECT MAX(id) from info").fetchall()[0][0]
        b = c.execute("SELECT MAX(batchId) from info").fetchall()[0][0]
        if a is None:
            a = 1
        if b is None:
            b = 0
        c.execute(
            f"INSERT into info(batchId, height , width, torn , not_clean, timestamp) VALUES({b+1}, {height1} , {width1}, {torn1}, {not_clean1}, datetime())")
        c.execute(
            f"INSERT into info(batchId, height , width, torn , not_clean, timestamp) VALUES({b+1}, {height2} , {width2}, {torn2}, {not_clean2}, datetime())")
        c.execute(
            f"INSERT into info(batchId, height , width, torn , not_clean, timestamp) VALUES({b+1}, {height3} , {width3}, {torn3}, {not_clean3}, datetime())")
        c.execute(
            f"INSERT into info(batchId, height , width, torn , not_clean, timestamp) VALUES({b+1}, {height4} , {width4}, {torn4}, {not_clean4}, datetime())")

        c.execute(
            f"INSERT into result(id , result , timestamp) VALUES({b+1}, '{result4}', datetime() )")

        for i in area1:
            c.execute(
                f"""INSERT into area(id , name , area , xmin , xmax, ymin , ymax , timestamp) VALUES ({a},'{ i["name"]}' , { i["area"]} , {i["xmin"]},{i["xmax"]} , {i["ymin"]} , {i["ymax"]} , datetime())""")
        for i in area2:
            c.execute(
                f"""INSERT into area(id , name , area , xmin , xmax, ymin , ymax , timestamp) VALUES ({a+1},'{ i["name"]}' , { i["area"]} , {i["xmin"]},{i["xmax"]} , {i["ymin"]} , {i["ymax"]} , datetime())""")
        for i in area3:
            c.execute(
                f"""INSERT into area(id , name , area , xmin , xmax, ymin , ymax , timestamp) VALUES ({a+2},'{ i["name"]}' , { i["area"]} , {i["xmin"]},{i["xmax"]} , {i["ymin"]} , {i["ymax"]} , datetime())""")
        for i in area4:
            c.execute(
                f"""INSERT into area(id , name , area , xmin , xmax, ymin , ymax , timestamp) VALUES ({a+3},'{ i["name"]}' , { i["area"]} , {i["xmin"]},{i["xmax"]} , {i["ymin"]} , {i["ymax"]} , datetime())""")

        conn.commit()
        conn.close()
        dbend = time.perf_counter()
        # print('database time =' ,  dbend - dbstart)

    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def get_top(self, image):
        dimstart = time.perf_counter()
        width = 3
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        # loop over the contours individually

        for c in cnts:

            if cv2.contourArea(c) < 100 and cv2.contourArea(c) > 20:
                continue

            # print(i)

            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(
                box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            box = perspective.order_points(box)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)

            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # USED FOR CALLIBERATION
            # if pixelsPerMetric is None:
            #pixelsPerMetric = dB / width

            pixelsPerMetric = 90.33
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            dimA = dimA*(25.4)
            dimB = dimB*(25.4)
            dimend = time.perf_counter()
            # print('get top time =' ,  dimend - dimstart)
            # print('get dimension time = ' , )
            return dimA, dimB

    def get_dimensions(self, image):
        dimstart = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # print("Done with Contours")

        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
        # print(len(c))
        for c in cnts:
            if cv2.contourArea(c) < 900:
                continue
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(
                box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # if pixelsPerMetric is None:
            #     pixelsPerMetric = dB / 5
            pixelsPerMetric = 61
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            dimend = time.perf_counter()
            # print('get dimension time =' ,  dimend - dimstart)
            # print('get dimension time = ' , )
            return dimA*25.4, dimB*25.4

    def getImages(self):
        timestart = time.perf_counter()

        print('calling getimages')

        self.accepted_list = []
        # self.loadingImg.destroy()


        # replace  btn
        self.myBtn.place(x=self.width*0.90, y=self.height*0.009)
        self.myBtn.config(width=6)
        self.myBtn = Button(
            text="Predict", command=self.callGetImages, style="TButton")




        # image reading logic here

        top = cv2.imread('1.jpg')
        bottom = cv2.imread('2.jpg')
        side1 = cv2.imread('5.jpeg')
        side2 = cv2.imread('6.jpeg')

        # with concurrent.futures.ThreadPoolExecutor() as executer:
        #     future = executer.submit(self.renderData, top , 'top')
        w1, h1, area1,  Torn1, notClean1, result1, img1 = self.renderData(
            top, 'top')
        w2, h2, area2,  Torn2, notClean2, result2, img2 = self.renderData(
            bottom, 'bottom')
        w3, h3, area3,  Torn3, notClean3, result3, img3 = self.renderData(
            side1, 'side1')
        w4, h4, area4,  Torn4, notClean4, result4, img4 = self.renderData(
            side2, 'side2')




        # creating directory for saving image for today
        dirname = f'C:/Users/wahab/Desktop/thread/img_database/{self.date}/'
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        # write images in folder
        cv2.imwrite(os.path.join(
            dirname, f'{self.count}.jpg'), img1)
        cv2.imwrite(os.path.join(
            dirname, f'{self.count+1}.jpg'), img2)
        cv2.imwrite(os.path.join(
            dirname, f'{self.count+2}.jpg'), img3)
        cv2.imwrite(os.path.join(
            dirname, f'{self.count+3}.jpg'), img4)



        timeend = time.perf_counter() - timestart
        # print('total time =',  timeend - timestart)
       

        if True in self.accepted_list:
            self.submit(height1=h1, width1=w1, torn1=Torn1, not_clean1=notClean1, area1=area1, result1="False",   height2=h2, width2=w2, torn2=Torn2, not_clean2=notClean2, area2=area2, result2="False",
                        height3=h3, width3=w3, torn3=Torn3, not_clean3=notClean3, area3=area3, result3="False",   height4=h4, width4=w4, torn4=Torn4, not_clean4=notClean4, area4=area4, result4="True")
            value_label = Label(
                self.resultFrame, text=f"Rejected , time = {timeend:.2f}", font=("Courier", 23), borderwidth=1, style="BW.TLabel", relief="solid",  padding=10, anchor="center", justify=CENTER
            )
            value_label.grid(row=5, column=1, sticky=N +
                             S+W+E, pady=self.height*0.02)
        else:
            self.submit(height1=h1, width1=w1, torn1=Torn1, not_clean1=notClean1, area1=area1, result1="False",   height2=h2, width2=w2, torn2=Torn2, not_clean2=notClean2, area2=area2, result2="False",
                        height3=h3, width3=w3, torn3=Torn3, not_clean3=notClean3, area3=area3, result3="False",   height4=h4, width4=w4, torn4=Torn4, not_clean4=notClean4, area4=area4, result4="False")
            value_label = Label(
                self.resultFrame, text=f"Accepted, time = {timeend:.2f}", font=("Courier", 23), borderwidth=1, style="BW.TLabel", relief="solid",  padding=10, anchor="center", justify=CENTER
            )
            value_label.grid(row=1, column=0, sticky=N +
                             S+W+E, pady=self.height*0.02)

        self.t1Check = True
        print(self.accepted_list)
        # for widget in status_frame.winfo_children() :   
        #     print(f'destroying {widget}')
        #     widget.destroy()


        # for widget in head_frame.winfo_children():
        #     print(f'destroying {widget}')
        #     widget.destroy()
        
        # for widget in self.winfo_children():
        #     print(f'destroying {widget}')
        #     widget.destroy()
        # time.sleep(0.2)
        self.getImages()
        


    def renderData(self, image, side):
        start = time.perf_counter()
        sb_text = 13
        info_text = 13
        im = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if side == 'top' or side == 'bottom':
            w, h = self.get_top(im)
            w, h = round(w, 2), round(h, 2)
        else:
            w, h = self.get_dimensions(im)
            w, h = round(w, 2), round(h, 2)

        # print("image shape , ",im.shape)
        img, area,  Torn, notClean, result = yoloTiny.detect(im)
        # style = ttk.Style()
        # style.configure("BW.TLabel", foreground="navy",
        #                 background="light blue", borderwidth=1, relief="solid")
        

        
        # add text
        tt = time.perf_counter()-start
        text1 = f"width = {w}\nheight = {h}\nnumber of torns={Torn}\nnumber of dirty={notClean}\ntime taken = {tt:.2f} seconds"

        # self.rframe.grid_columnconfigure(0, weight=1,)
        img1 = cv2.resize(img, (int(self.width*0.11), int(self.height*0.16)))
        img1 = PIL.Image.fromarray(img1)
        img1 = ImageTk.PhotoImage(img1)
    


        if side=='top':
            self.label41.configure(image=img1) 
            self.label41.image = img1
            self.label41.grid(row=2, column=0, columnspan=1,sticky="", pady=self.height*0.01)
            head1 = Label(self.rframe1, text=f"{side} view", font=(
            "Courier", sb_text), style="BW.TLabel", padding=5)
            head1.grid(row=0, column=0, sticky=N+S+W+E)
            t1_label = Label(self.rframe1, text=text1, font=("Courier", info_text), style="BW.TLabel", padding=5)
            t1_label.grid(row=1, column=0, sticky=N+S+W+E,pady=self.height*0.01, padx=self.width*0.02)
            
        if side=='bottom':
            self.label42.configure(image=img1) 
            self.label42.image = img1
            self.label42.grid(row=3, column=0, columnspan=1,
            sticky="", pady=self.height*0.01)
            head1 = Label(self.rframe1, text=f"{side} view", font=(
            "Courier", sb_text), style="BW.TLabel", padding=5)
            head1.grid(row=2, column=0, sticky=N+S+W+E)
            t1_label = Label(self.rframe1, text=text1, font=("Courier", info_text), style="BW.TLabel", padding=5)
            t1_label.grid(row=3, column=0, sticky=N+S+W+E,pady=self.height*0.01, padx=self.width*0.02)
        if side=='side1':
            self.label43.configure(image=img1) 
            self.label43.image = img1
            self.label43.grid(row=4, column=0, columnspan=1,
            sticky="", pady=self.height*0.01)
            head1 = Label(self.rframe1, text=f"{side} view", font=(
            "Courier", sb_text), style="BW.TLabel", padding=5)
            head1.grid(row=4, column=0, sticky=N+S+W+E)
            t1_label = Label(self.rframe1, text=text1, font=("Courier", info_text), style="BW.TLabel", padding=5)
            t1_label.grid(row=5, column=0, sticky=N+S+W+E,pady=self.height*0.01, padx=self.width*0.02)
        if side=='side2':
            self.label44.configure(image=img1) 
            self.label44.image = img1
            self.label44.grid(row=5, column=0, columnspan=1,
            sticky="", pady=self.height*0.01)
            head1 = Label(self.rframe1, text=f"{side} view", font=(
            "Courier", sb_text), style="BW.TLabel", padding=5)
            head1.grid(row=6, column=0, sticky=N+S+W+E)
            t1_label = Label(self.rframe1, text=text1, font=("Courier", info_text), style="BW.TLabel", padding=5)
            t1_label.grid(row=7, column=0, sticky=N+S+W+E,pady=self.height*0.01, padx=self.width*0.02)


        # label4.place(x=pos[i-1]['x'], y=pos[i-1]['y'])
        # if side == 'top':


        # elif side == 'bottom':


        # if side == 'side1':


        # if side == 'side2':


        self.accepted_list.append(result)
        return w, h, area,  Torn, notClean, result, img

    def quit_and_report(self):
        conn = sqlite3.connect(
            'C:/Users/wahab/Desktop/thread/thread.db')
        c = conn.cursor()

        accepted = c.execute(
            "SELECT COUNT(*) from result where result='False' and date(timestamp) == date(datetime()) ").fetchall()[0][0]
        rejected = c.execute(
            "SELECT COUNT(*) from result where result='True' and  date(timestamp) == date(datetime()) ").fetchall()[0][0]
        total_batches = c.execute(
            "SELECT COUNT(DISTINCT batchId) from info where date(timestamp) == date(datetime()) ").fetchall()[0][0]
        batches = c.execute(
            "SELECT torn , not_clean , timestamp from info where date(timestamp) == date(datetime())  ").fetchall()
        total_correct_dimensions = c.execute(
            "SELECT COUNT(DISTINCT batchId)  from info where date(timestamp) == date(datetime()) and height>=20 and height<=25  ").fetchall()[0][0]
        total_wrong_dimensions = total_batches - total_correct_dimensions
        total_torn = 0
        total_dirty = 0

        # print(b)
        for a in batches:
            total_torn += a[0]
            total_dirty += a[1]

        conn.commit()
        conn.close()

        # Create a workbook and add a worksheet.
        workbook = xlsxwriter.Workbook(
            f'C:/Users/wahab/Desktop/thread/daily_reports/{self.date}.xlsx')
        worksheet = workbook.add_worksheet()

        worksheet.set_column('A:A', 40)
        # Insert an image.

        worksheet.insert_image('A1', 'logo2.png')

        # Some data we want to write to the worksheet.
        expenses = (
            ['total tested ', total_batches],
            ['total accepted ',   accepted],
            ['total rejected ',  rejected],
            ['total number of torns',    total_torn],
            ['total number of dirty',    total_dirty],
            ['average number of torns per loom',    total_torn/4],
            ['average number of dirty per loom',    total_dirty/4],
            ['total correct dimensions',    total_correct_dimensions],
            ['total wrong dimensions',    total_wrong_dimensions],
        )

        # Start from the first cell. Rows and columns are zero indexed.
        row = 11
        col = 0
        bold = workbook.add_format({'bold': True})
        bold.set_font_size(15)
        worksheet.write(10, 0, 'Date and Time started', bold)
        worksheet.write(10, 1, str(self.date) + '  '+str(self.time_started))
        # Iterate over the data and write it out row by row.
        for item, cost in (expenses):
            worksheet.write(row, col,     item, bold)
            worksheet.write(row, col + 1, cost)
            row += 1

        workbook.close()

        self.master.destroy()

    def callGetImages(self):
        self.startBtnCheck = True
        self.t1Check = False
        self.t1 = threading.Thread(target=self.getImages).start()
        if self.t1Check:
            self.t1.join()
            # print('thread killed')

    def initUI(self):
        self.startBtnCheck= False
        self.count = 1
        self.master.title("Thread Analysis")
        self.pack(fill=BOTH, expand=1)
        # lImg = PIL.Image.open('logo2.png')
        # lImg = ImageTk.PhotoImage(lImg)
        # self.loadingImg = Label(self, image=lImg)
        # self.loadingImg.image = lImg
        # self.loadingImg.place(x=self.width/2 - 190, y=self.height/2 - 150)

        self.date = datetime.date.today()

        self.time_started = datetime.datetime.now().strftime("%I:%M:%S %p")

        # self.getImages()

        
        # style = Style()
        # style.theme_use('breeze')
        # style.configure("TButton", foreground="red",
        #                 background="light sky blue",  font=('Helvetica', 22))

        self.myBtn = Button(
            text="Predict", command=self.callGetImages, style="TButton")

        self.myBtn.place(x=self.width/2.35, y=self.height-290)
        self.myBtn.config(width=20)

        style1 = ttk.Style()
        style1.configure("TLabel", foreground="red", background="light blue")

        self.head_frame = Frame(self)
        self.head_frame.grid(row=0, column=0, columnspan=3,
                        sticky="", pady=self.height*0.009)

        self.main_head = Label(self.head_frame, text="THREAD CONE QA SYSTEM",
                          font=("Courier", 25))

        self.main_head.grid(row=0, column=1, sticky="")
        # self.grid_columnconfigure(0, weight=1)
      

        # USING FOR HEADING 1
        style = ttk.Style()
        style.configure("BW1.TLabel", foreground="navy", background="light blue")


      
              # Stye for Headiing3
        # style = ttk.Style()
        # style.configure("BW1.TLabel", foreground="navy",
        #                 background="light blue")
        self.head1 = Label(self.head_frame, text=f"CURRENT CASE", font=("Courier", 19), style="BW1.TLabel",  justify=CENTER ,padding=10)
        self.head2 = Label(self.head_frame, text=f"TODAY SUMMARY", font=("Courier", 19), style="BW1.TLabel", justify=CENTER , padding=10)
        self.head3 = Label(self.head_frame, text="CURRENT RESULT", font=("Courier", 19), style="BW.TLabel", borderwidth=0,anchor="center", justify=CENTER , padding=10)
        # status_label.grid(row=1, column=5,)

        self.head1.grid(row=1, column=1, columnspan=1, sticky="")
        # self.head2.grid(row=1, column=20, columnspan=1, sticky="")
        self.head3.grid(row=1, column=10, columnspan=1, sticky="")
        # self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure(2, weight=1)


        # if self.startBtnCheck:
        quitBtn = Button(
            text="Quit", command=self.quit_and_report, style="TButton")
        quitBtn.place(x=self.width*0.90, y=self.height*0.069)
        quitBtn.config(width=6)


        self.resultFrame = Frame(self)
        self.resultFrame.grid(row=2, column=40, columnspan=1, sticky=N +
            S+W+E, pady=self.height*0.01, padx=self.width*0.02)
        # summary data =========================== ad data here
        conn = sqlite3.connect(
            'C:/Users/wahab/Desktop/thread/thread.db')
        c = conn.cursor()

        accepted = c.execute(
            "SELECT COUNT(*) from result where result='False'  and date(timestamp) == date(datetime())").fetchall()[0][0]
        rejected = c.execute(
            "SELECT COUNT(*) from result where result='True'  and date(timestamp) == date(datetime())").fetchall()[0][0]
        total_batches = c.execute(
            "SELECT COUNT(DISTINCT batchId) from info  where date(timestamp) == date(datetime()) ").fetchall()[0][0]
        self.count = c.execute(
            "SELECT COUNT(id) from info ").fetchall()[0][0] +1

        style = ttk.Style()
        style.configure("BW.TLabel", foreground="navy",
                        background="light blue", borderwidth=1, relief="solid")

        text1 = f"Date = {self.date}\nTime Started = {self.time_started}\nTotal Accepted = {accepted}\nTotal Rejected = {rejected}\nTotal Batched Processed = {total_batches}\n"
        t1_label = Label(
            self.resultFrame, text=text1, font=("Courier", 14), style="BW.TLabel", padding=20
        )
        t1_label.grid(row=2, column=1, columnspan=1, rowspan=2,
                      sticky=N+S+W+E, pady=self.height*0.02, padx=self.width*0.03)

        conn.commit()
        conn.close()

        
        # self.status_frame = Frame(self, )
        # self.status_frame.grid(row=4, column=2, columnspan=1, rowspan=2,
        #                   sticky=N+S+W+E, pady=self.height*0.02, padx=self.width*0.03)


        # self.status_frame.grid_columnconfigure(0, weight=1)
        # self.status_frame.grid_rowconfigure(1, weight=1)


        self.rframe1 = Frame(self)
        self.rframe2 = Frame(self)
        self.rframe1.grid(row=2, column=1, columnspan=1, sticky=N +
            S+W+E, pady=self.height*0.01, padx=self.width*0.02)
        self.rframe2.grid(row=2, column=0, columnspan=1, sticky=N +
            S+W+E, pady=self.height*0.01, padx=self.width*0.02)

        self.label41 = Label(self.rframe2)
        self.label42 = Label(self.rframe2)
        self.label43 = Label(self.rframe2)
        self.label44 = Label(self.rframe2)
        # while True:
        #     self.getImages()


def main():
    conn = sqlite3.connect(
        'C:/Users/wahab/Desktop/thread/thread.db')
    c = conn.cursor()

    # creating table if not exits already

    c.execute("""CREATE TABLE IF NOT EXISTS info(id INTEGER PRIMARY KEY AUTOINCREMENT ,batchId INTEGER, height REAL ,width REAL, torn INTEGER , not_clean INTEGER ,timestamp TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS area(id INTEGER  ,name TEXT ,  area INTEGER , xmin INTEGER , xmax INTEGER ,ymin INTEGER , ymax INTEGER ,timestamp TEXT) """)
    c.execute(
        """CREATE TABLE IF NOT EXISTS result(id INTEGER PRIMARY KEY  , result TEXT ,timestamp TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS user(id INTEGER PRIMARY KEY  AUTOINCREMENT, username TEXT ,password TEXT)""")
    # c.execute("""INSERT INTO USER(username, password) values('wahab' , 'wahab123')""")

    # checking if record of two weeks old exists and make report of it and delete

    two_week_old = c.execute(
        """SELECT * FROM info WHERE timestamp < datetime('now', '-14 days');""").fetchall()
    if len(two_week_old) != 0:
        last_week_record = c.execute(
            """SELECT * FROM info WHERE timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days');""").fetchall()
        # print(last_week_record)
        accepted = c.execute(
            "SELECT COUNT(*) from result where result='False' and timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days') ").fetchall()[0][0]
        rejected = c.execute(
            "SELECT COUNT(*) from result where result='True' and  timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days') ").fetchall()[0][0]
        total_batches = c.execute(
            "SELECT COUNT(DISTINCT batchId) from info where timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days') ").fetchall()[0][0]
        batches = c.execute(
            "SELECT torn , not_clean , timestamp from info where timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days')  ").fetchall()
        total_correct_dimensions = c.execute(
            "SELECT COUNT(DISTINCT batchId)  from info where  height>=20 and height<=25  and timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days') ").fetchall()[0][0]
        total_wrong_dimensions = total_batches = total_correct_dimensions
        total_torn = 0
        total_dirty = 0

        # print(b)
        for a in batches:
            total_torn += a[0]
            total_dirty += a[1]

        # counting numbers of files in weekly reports for numbering
        DIR = "C:/Users/wahab/Desktop/thread/weekly_reports"
        week = len([name for name in os.listdir(DIR)
                    if os.path.isfile(os.path.join(DIR, name))])

        workbook = xlsxwriter.Workbook(
            f'C:/Users/wahab/Desktop/thread/weekly_reports/week-{week+1}.xlsx')
        worksheet = workbook.add_worksheet()

        worksheet.set_column('A:A', 40)
        # Insert an image.

        worksheet.insert_image('A1', 'logo2.png')

        # Some data we want to write to the worksheet.
        expenses = (
            ['total tested ', total_batches],
            ['total accepted ',   accepted],
            ['total rejected ',  rejected],
            ['total number of torns',    total_torn],
            ['total number of dirty',    total_dirty],
            ['average number of torns per loom',    total_torn/4],
            ['average number of dirty per loom',    total_dirty/4],
            ['total correct dimensions',    total_correct_dimensions],
            ['total wrong dimensions',    total_wrong_dimensions],
        )

        # Start from the first cell. Rows and columns are zero indexed.
        row = 11
        col = 0
        bold = workbook.add_format({'bold': True})
        bold.set_font_size(15)
        # Iterate over the data and write it out row by row.
        for item, cost in (expenses):
            worksheet.write(row, col,     item, bold)
            worksheet.write(row, col + 1, cost)
            row += 1

        workbook.close()

        c.execute(
            "DELETE FROM result where  timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days') ").fetchall()[0][0]
        c.execute(
            "DELETE FROM info where  timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days') ").fetchall()[0][0]
        c.execute(
            "DELETE FROM area where  timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days') ").fetchall()[0][0]

    conn.commit()
    conn.close()

    # checking if images in directory are two weeks old if yes then delete

    now = time.time()
    dir = 'C:/Users/wahab/Desktop/thread/img_database/'
    for f in os.listdir(dir):
        if os.stat(os.path.join(dir, f)).st_mtime < now - 14 * 86400:
            shutil.rmtree(os.path.join(dir, f))

    root = ThemedTk(theme="adapta")
    root.state('iconic')
    root.resizable(False, False)
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry(f"{width}x{height}")

    canvas = Canvas(root, width=width, height=height)

    app = Example(width, height, canvas)
    root.mainloop()


if __name__ == '__main__':
    load_model()
    main()