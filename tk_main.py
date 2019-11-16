#!/usr/bin/env python
#coding:utf-8
import numpy as np
from tkinter import *
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.filedialog import askopenfilename
import pandas as pd
from neural import Neural


class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.root = master
        self.grid()
        self.create_widgets()
    
    def create_widgets(self):

        self.winfo_toplevel().title("Yochien NN HW1")

        # 設定訓練循環次數
        self.learning_rate_label = tk.Label(self)
        self.learning_rate_label["text"] = "學習率"
        self.learning_rate_label.grid(row=0, column=0, sticky=tk.N+tk.W)

        self.learning_rate = tk.DoubleVar()
        self.learning_rate_entry = tk.Entry(self, textvariable=self.learning_rate)
        self.learning_rate_entry.grid(row=0, column=1, sticky=tk.N+tk.W)

        # 設定訓練循環次數
        self.epoch_label = tk.Label(self)
        self.epoch_label["text"] = "訓練循環次數"
        self.epoch_label.grid(row=1, column=0, sticky=tk.N+tk.W)

        self.epoch_spinbox = tk.Spinbox(self, from_=1, to=1000)
        self.epoch_spinbox.grid(row=1, column=1, sticky=tk.N+tk.W)
        
        # 設定提前結束條件
        self.early_stop_label = tk.Label(self)
        self.early_stop_label["text"] = "提前結束訓練正確率(%)"
        self.early_stop_label.grid(row=2, column=0, sticky=tk.N+tk.W)

        self.early_stop_spinbox = tk.Spinbox(self, from_=1, to=100)
        self.early_stop_spinbox.grid(row=2, column=1, sticky=tk.N+tk.W)

        # 選取資料集檔案
        self.load_data_label = tk.Label(self)
        self.load_data_label["text"] = "載入資料集"
        self.load_data_label.grid(row=3, column=0, sticky=tk.N+tk.W)

        self.load_data_button = tk.Button(self)
        self.load_data_button["text"] = "選取檔案"
        self.load_data_button.grid(row=3, column=1, sticky=tk.N+tk.W)
        self.load_data_button["command"] = self.select_file

        self.file_path_label = tk.Label(self)
        self.file_path_label["text"] = ""
        self.file_path_label.grid(row=3, column=2, sticky=tk.N+tk.W)

        # run run
        self.run_label = tk.Label(self)
        self.run_label["text"] = "進行訓練及測試"
        self.run_label.grid(row=4, column=0, sticky=tk.N+tk.W)

        self.run_button = tk.Button(self)
        self.run_button["text"] = "Run"
        self.run_button.grid(row=4, column=1, sticky=tk.N+tk.W)
        self.run_button["command"] = self.run


        # 設定訓練圖表
        self.training_acc_figure = Figure(figsize=(4,4), dpi=100)
        self.training_acc_canvas = FigureCanvasTkAgg(self.training_acc_figure, self)
        self.training_acc_canvas.draw()
        self.training_acc_canvas.get_tk_widget().grid(row=5, column=0, columnspan=3)

        self.training_data_figure = Figure(figsize=(4,4), dpi=100)
        self.training_data_canvas = FigureCanvasTkAgg(self.training_data_figure, self)
        self.training_data_canvas.draw()
        self.training_data_canvas.get_tk_widget().grid(row=5, column=4, columnspan=3)

        # 相關結果文字
        self.training_epoch_label = tk.Label(self)
        self.training_epoch_label["text"] = "實際訓練次數(Epoch)"
        self.training_epoch_label.grid(row=6, column=0, sticky=tk.N+tk.W)

        self.training_epoch_text_label = tk.Label(self)
        self.training_epoch_text_label["text"] = ""
        self.training_epoch_text_label.grid(row=6, column=1, sticky=tk.N+tk.W)

        self.training_acc_label = tk.Label(self)
        self.training_acc_label["text"] = "訓練辨識率(%)"
        self.training_acc_label.grid(row=7, column=0, sticky=tk.N+tk.W)

        self.training_acc_text_label = tk.Label(self)
        self.training_acc_text_label["text"] = ""
        self.training_acc_text_label.grid(row=7, column=1, sticky=tk.N+tk.W)

        self.testing_acc_label = tk.Label(self)
        self.testing_acc_label["text"] = "測試辨識率(%)"
        self.testing_acc_label.grid(row=8, column=0, sticky=tk.N+tk.W)

        self.testing_acc_text_label = tk.Label(self)
        self.testing_acc_text_label["text"] = ""
        self.testing_acc_text_label.grid(row=8, column=1, sticky=tk.N+tk.W)

        self.weight_label = tk.Label(self)
        self.weight_label["text"] = "當前鍵結值"
        self.weight_label.grid(row=9, column=0, sticky=tk.N+tk.W)

        self.weight_text = tk.Text(self)
        self.weight_text["height"] = 5
        self.weight_text["width"] = 40
        self.weight_text.grid(row=9, column=1, sticky=tk.N+tk.W)

        self.bias_label = tk.Label(self)
        self.bias_label["text"] = "當前偏差(Bias)"
        self.bias_label.grid(row=10, column=0, sticky=tk.N+tk.W)

        self.bias_text_label = tk.Label(self)
        self.bias_text_label["text"] = ""
        self.bias_text_label.grid(row=10, column=1, sticky=tk.N+tk.W)

    def draw_training_acc_figure(self, train_result_list):
        #清空影像
        self.training_acc_figure.clf()
        self.training_acc_figure.a = self.training_acc_figure.add_subplot(111)
        acc_list = [i['acc'] for i in train_result_list]

        #使y範圍為0-105
        self.training_acc_figure.a.set_ylim([0, 105])

        #繪製正確率折線圖
        self.training_acc_figure.a.plot(acc_list)
        self.training_acc_figure.a.set_title('Traing: Acc/Epoch')
        self.training_acc_canvas.draw()


    def draw_training_data_figure(self, dataset, testing_dataset, last_train_result):
        # 清空影像
        self.training_data_figure.clf()
        self.training_data_figure.a = self.training_data_figure.add_subplot(111)

        # 產生全部資料 X,y list
        X_0 = dataset[dataset[len(dataset.columns)-1]==0][0].values.reshape(-1,).tolist()
        y_0 = dataset[dataset[len(dataset.columns)-1]==0][1].values.reshape(-1,).tolist()
        X_1 = dataset[dataset[len(dataset.columns)-1]==1][0].values.reshape(-1,).tolist()
        y_1 = dataset[dataset[len(dataset.columns)-1]==1][1].values.reshape(-1,).tolist()
        
        # draw 全部資料集兩種分類資料的點位
        self.training_data_figure.a.plot(X_0, y_0, 'ro')
        self.training_data_figure.a.plot(X_1, y_1, 'bo')

        # 產生測試資料資料 X,y list
        X_test = testing_dataset[0].values.reshape(-1,).tolist()
        y_test = testing_dataset[1].values.reshape(-1,).tolist()

        # draw 測試資料的點位
        self.training_data_figure.a.plot(X_test, y_test, 'wx')

        # 保存全部資料集的畫布範圍
        xmin = self.training_data_figure.a.get_xlim()[0]
        xmax = self.training_data_figure.a.get_xlim()[1]
        ymin = self.training_data_figure.a.get_ylim()[0]
        ymax = self.training_data_figure.a.get_ylim()[1]

        # 感知機一次方程式
        wx = float(last_train_result["weight"][0])
        wy = float(last_train_result["weight"][1])
        bias = float(last_train_result["bias"])
        result_x = np.linspace(-100, 100, 100)  
        result_y = - (result_x * wx  - bias) / wy         

        # draw 感知機一次方程式(form -100~100)
        self.training_data_figure.a.plot(result_x, result_y)    
        
        # draw 還原全部資料集的畫布範圍
        self.training_data_figure.a.set_xlim([xmin,xmax])
        self.training_data_figure.a.set_ylim([ymin,ymax])

        self.training_data_figure.a.set_title('Traing Data')
        self.training_data_canvas.draw()
          

    def select_file(self):
        try:
            filename = askopenfilename()
            self.file_path_label["text"] = filename
        except Exception as e:
            print(e)
            self.file_path_label["text"] = ""
        
    def run(self):
        filename = self.file_path_label["text"]
        if(filename == ""):
            print("未選取檔案")
            tk.messagebox.showinfo("Error","未選取資料集")
            return
        df = pd.read_table(filename, sep=" ", header=None)

        # 檢查是否二類問題
        df_label = np.split(df, [len(df.columns)-1], axis=1)[1]
        if(len(df_label.groupby(len(df.columns)-1).groups) != 2):
            print("非二類問題")
            tk.messagebox.showinfo("Error","資料集非二類問題")
            return 
            # 非二類問題
        # label非0/1組合 改變label-> 0~1
        if (0 not in df_label.groupby(len(df.columns)-1).groups) or (1 not in df_label.groupby(len(df.columns)-1).groups):
            df[len(df.columns)-1] = df[len(df.columns)-1] % 2
       
        # split traning data and testing data
        train_df=df.sample(frac=0.666666)
        test_df=df.drop(train_df.index)

        train_dfs = np.split(train_df, [len(train_df.columns)-1], axis=1)
        train_X_df = train_dfs[0]
        train_y_df = train_dfs[1]
        train_X = train_X_df.values.tolist()
        train_y = train_y_df.values.reshape(-1,).tolist()

        test_dfs = np.split(test_df, [len(test_df.columns)-1], axis=1)
        test_X_df = test_dfs[0]
        test_y_df = test_dfs[1]
        test_X = test_X_df.values.tolist()
        test_y = test_y_df.values.reshape(-1,).tolist()

        #learning_rate = 0.8
        learning_rate = self.learning_rate.get()

        # run training and show result
        n = Neural(train_X, train_y, learning_rate)
        train_result_list = []
        print("### training start ###")
        for i in range(int(self.epoch_spinbox.get())):
            self.training_epoch_text_label["text"] = i + 1
            train_result = n.train()
            train_result_list.append(train_result)
            if train_result["acc"] > float(self.early_stop_spinbox.get()):
                break
        print("### training end ###")
        self.draw_training_acc_figure(train_result_list)
        self.training_acc_text_label["text"] = train_result_list[len(train_result_list)-1]["acc"]
        self.bias_text_label["text"] = train_result_list[len(train_result_list)-1]["bias"]
        self.weight_text.delete(1.0, END) 
        self.weight_text.insert(1.0, train_result_list[len(train_result_list)-1]["weight"]) 
        
        # run testing and show result
        print("### predict start ###")
        test_result = n.test(test_X, test_y)
        print("### predict end ###")

        self.testing_acc_text_label["text"] = test_result["acc"]

        # draw training data and predict line
        self.draw_training_data_figure(df, test_df, train_result_list[len(train_result_list)-1])


root = tk.Tk()
app = Application(root)
root.mainloop()