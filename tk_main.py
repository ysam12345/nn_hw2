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
from mlp import MLP


class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.root = master
        self.grid()
        self.create_widgets()
        self.draw_original_figure = True
        self.number_classify_data_index = 0

    def create_widgets(self):

        self.winfo_toplevel().title("Yochien NN HW2")

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

        # 設定隱藏層層數
        self.hidden_layer_num_label = tk.Label(self)
        self.hidden_layer_num_label["text"] = "隱藏層層數"
        self.hidden_layer_num_label.grid(row=3, column=0, sticky=tk.N+tk.W)

        self.hidden_layer_num_spinbox = tk.Spinbox(self, from_=1, to=100)
        self.hidden_layer_num_spinbox.grid(row=3, column=1, sticky=tk.N+tk.W)

        # 設定隱藏層神經元數量
        self.hidden_layer_neural_num_label = tk.Label(self)
        self.hidden_layer_neural_num_label["text"] = "隱藏層神經元數量"
        self.hidden_layer_neural_num_label.grid(row=4, column=0, sticky=tk.N+tk.W)

        self.hidden_layer_neural_num_spinbox = tk.Spinbox(self, from_=1, to=100)
        self.hidden_layer_neural_num_spinbox.grid(row=4, column=1, sticky=tk.N+tk.W)

    
        # 選取資料集檔案
        self.load_data_label = tk.Label(self)
        self.load_data_label["text"] = "載入資料集"
        self.load_data_label.grid(row=5, column=0, sticky=tk.N+tk.W)

        self.load_data_button = tk.Button(self)
        self.load_data_button["text"] = "選取檔案"
        self.load_data_button.grid(row=5, column=1, sticky=tk.N+tk.W)
        self.load_data_button["command"] = self.select_file

        self.file_path_label = tk.Label(self)
        self.file_path_label["text"] = ""
        self.file_path_label.grid(row=5, column=2, sticky=tk.N+tk.W)

        # run run
        self.run_label = tk.Label(self)
        self.run_label["text"] = "進行訓練及測試"
        self.run_label.grid(row=6, column=0, sticky=tk.N+tk.W)

        self.run_button = tk.Button(self)
        self.run_button["text"] = "Run"
        self.run_button.grid(row=6, column=1, sticky=tk.N+tk.W)
        self.run_button["command"] = self.run

        # 切換圖表顯示(原始資料/預測結果)
        self.change_training_data_figure_label = tk.Label(self)
        self.change_training_data_figure_label["text"] = "切換圖表顯示(原始資料/預測結果)"
        self.change_training_data_figure_label.grid(row=7, column=0, sticky=tk.N+tk.W)

        self.change_training_data_figure_button = tk.Button(self)
        self.change_training_data_figure_button["text"] = "切換"
        self.change_training_data_figure_button.grid(row=7, column=1, sticky=tk.N+tk.W)
        self.change_training_data_figure_button["command"] = self.change_training_data_figure

        self.training_data_figure_type_label = tk.Label(self)
        self.training_data_figure_type_label["text"] = "Original"
        self.training_data_figure_type_label.grid(row=7, column=2, sticky=tk.N+tk.W)

        # 設定訓練圖表
        self.training_acc_figure = Figure(figsize=(4,4), dpi=100)
        self.training_acc_canvas = FigureCanvasTkAgg(self.training_acc_figure, self)
        self.training_acc_canvas.draw()
        self.training_acc_canvas.get_tk_widget().grid(row=8, column=0, columnspan=3)

        self.training_data_figure = Figure(figsize=(4,4), dpi=100)
        self.training_data_canvas = FigureCanvasTkAgg(self.training_data_figure, self)
        self.training_data_canvas.draw()
        self.training_data_canvas.get_tk_widget().grid(row=8, column=4, columnspan=3)

        # 相關結果文字
        self.training_epoch_label = tk.Label(self)
        self.training_epoch_label["text"] = "實際訓練次數(Epoch)"
        self.training_epoch_label.grid(row=9, column=0, sticky=tk.N+tk.W)

        self.training_epoch_text_label = tk.Label(self)
        self.training_epoch_text_label["text"] = ""
        self.training_epoch_text_label.grid(row=9, column=1, sticky=tk.N+tk.W)

        self.training_acc_label = tk.Label(self)
        self.training_acc_label["text"] = "訓練辨識率(%)"
        self.training_acc_label.grid(row=10, column=0, sticky=tk.N+tk.W)

        self.training_acc_text_label = tk.Label(self)
        self.training_acc_text_label["text"] = ""
        self.training_acc_text_label.grid(row=10, column=1, sticky=tk.N+tk.W)

        self.testing_acc_label = tk.Label(self)
        self.testing_acc_label["text"] = "測試辨識率(%)"
        self.testing_acc_label.grid(row=11, column=0, sticky=tk.N+tk.W)

        self.testing_acc_text_label = tk.Label(self)
        self.testing_acc_text_label["text"] = ""
        self.testing_acc_text_label.grid(row=11, column=1, sticky=tk.N+tk.W)

        self.hidden_layer_weight_label = tk.Label(self)
        self.hidden_layer_weight_label["text"] = "隱藏層鍵結值"
        self.hidden_layer_weight_label.grid(row=12, column=0, sticky=tk.N+tk.W)

        self.hidden_layer_weight_text = tk.Text(self)
        self.hidden_layer_weight_text["height"] = 5
        self.hidden_layer_weight_text["width"] = 40
        self.hidden_layer_weight_text.grid(row=12, column=1, sticky=tk.N+tk.W)

        self.output_layer_weight_label = tk.Label(self)
        self.output_layer_weight_label["text"] = "輸出層鍵結值"
        self.output_layer_weight_label.grid(row=13, column=0, sticky=tk.N+tk.W)

        self.output_layer_weight_text = tk.Text(self)
        self.output_layer_weight_text["height"] = 5
        self.output_layer_weight_text["width"] = 40
        self.output_layer_weight_text.grid(row=13, column=1, sticky=tk.N+tk.W)

        self.number_classify_label = tk.Label(self)
        self.number_classify_label["text"] = "數字辨識結果"
        self.number_classify_label.grid(row=14, column=0, sticky=tk.N+tk.W)

        self.number_classify_data_label = tk.Label(self)
        self.number_classify_data_label["text"] = ""
        self.number_classify_data_label.grid(row=14, column=1, sticky=tk.N+tk.W)

        self.change_number_classify_data_button = tk.Button(self)
        self.change_number_classify_data_button["text"] = "切換數字辨識資料"
        self.change_number_classify_data_button.grid(row=14, column=2, sticky=tk.N+tk.W)
        self.change_number_classify_data_button["command"] = self.change_number_classify_data

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


    def draw_original_training_data_figure(self, dataset, testing_dataset):
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

        self.training_data_figure.a.set_title('Traing Data(Original)')
        self.training_data_canvas.draw()

    def draw_predict_training_data_figure(self, predict_result, testing_dataset):
        # 清空影像
        self.training_data_figure.clf()
        self.training_data_figure.a = self.training_data_figure.add_subplot(111)

        dataset = pd.DataFrame()
        for output in predict_result["output_list"]:
            d = output["input"]
            d.append(output["predict"])
            dataset = dataset.append([d])

        label_df = np.split(dataset, [len(dataset.columns)-1], axis=1)[1].values.reshape(-1,).tolist()
        label_list = list(set(label_df))

        # 產生全部資料 X,y list
        X_0 = dataset[dataset[len(dataset.columns)-1]==label_list[0]][0].values.reshape(-1,).tolist()
        y_0 = dataset[dataset[len(dataset.columns)-1]==label_list[0]][1].values.reshape(-1,).tolist()
        X_1 = dataset[dataset[len(dataset.columns)-1]==label_list[1]][0].values.reshape(-1,).tolist()
        y_1 = dataset[dataset[len(dataset.columns)-1]==label_list[1]][1].values.reshape(-1,).tolist()
        
        # draw 全部資料集兩種分類資料的點位
        self.training_data_figure.a.plot(X_0, y_0, 'co')
        self.training_data_figure.a.plot(X_1, y_1, 'go')

        # 產生測試資料資料 X,y list
        X_test = testing_dataset[0].values.reshape(-1,).tolist()
        y_test = testing_dataset[1].values.reshape(-1,).tolist()

        # draw 測試資料的點位
        self.training_data_figure.a.plot(X_test, y_test, 'wx')

        self.training_data_figure.a.set_title('Traing Data(Predict)')
        self.training_data_canvas.draw()

    def change_training_data_figure(self):
        print(self.draw_original_figure)
        if self.draw_original_figure == True:
            self.draw_original_figure = False
            self.training_data_figure_type_label["text"] = "Predict"
            self.draw_predict_training_data_figure(self.predict_result, self.test_df)
        else:
            self.draw_original_figure = True
            self.training_data_figure_type_label["text"] = "Original"
            self.draw_original_training_data_figure(self.df, self.test_df)

    def change_number_classify_data(self):
        self.number_classify_data_index += 1
        if self.number_classify_data_index >= self.data_num:
            self.number_classify_data_index = 0
        num_str = ""
        num_list = self.predict_result["output_list"][self.number_classify_data_index]["input"]
        for i in range(5):
            for j in range(5):
                num_str += str(int(num_list[ 5*i + j ]))
            num_str += "\n"
        self.number_classify_data_label["text"] = num_str
        self.number_classify_label["text"] = "數字辨識結果： " + str(self.category_map[self.predict_result["output_list"][self.number_classify_data_index]["predict"]]) + "\n 實際值： " + str(self.category_map[self.predict_result["output_list"][self.number_classify_data_index]["expect"]])

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

        # 計算label數量
        label_df = np.split(df, [len(df.columns)-1], axis=1)[1].values.reshape(-1,).tolist()
        label_set = set(label_df)
        category_num = len(label_set)
        # 改變label至[0, 1]範圍
        category_list = np.arange(0, 1, 1/(category_num - 1) ).tolist()
        category_list.append(1.0)
        i = 0
        self.category_map = {}
        for origin_label in label_set:
            self.category_map[category_list[i]] = origin_label
            df.loc[df[len(df.columns)-1] == origin_label ,len(df.columns)-1] = category_list[i]
            i += 1

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
        hidden_layer_num = int(self.hidden_layer_num_spinbox.get())
        hidden_layer_neural_num = int(self.hidden_layer_neural_num_spinbox.get())

        # run training and show result
        mlp = MLP(train_X, train_y, learning_rate, category_num, hidden_layer_num, hidden_layer_neural_num)
        train_result_list = []
        print("### training start ###")
        for i in range(int(self.epoch_spinbox.get())):
            self.training_epoch_text_label["text"] = i + 1
            train_result = mlp.train()
            train_result_list.append(train_result)
            #self.draw_training_data_figure(df, test_df, train_result_list[len(train_result_list)-1])
            if train_result["acc"] > float(self.early_stop_spinbox.get()):
                break
        print("### training end ###")
        self.draw_training_acc_figure(train_result_list)
        self.training_acc_text_label["text"] = train_result_list[len(train_result_list)-1]["acc"]
        self.hidden_layer_weight_text.delete(1.0, END) 
        self.hidden_layer_weight_text.insert(1.0, train_result_list[len(train_result_list)-1]["weight"]["hidden_layer_weight"]) 
        self.output_layer_weight_text.delete(1.0, END) 
        self.output_layer_weight_text.insert(1.0, train_result_list[len(train_result_list)-1]["weight"]["output_layer_weight"]) 
        
        # run testing and show result
        print("### predict start ###")
        test_result = mlp.test(test_X, test_y)
        print("### predict end ###")

        self.testing_acc_text_label["text"] = test_result["acc"]

        # draw training data and create predict df
        self.df = df
        self.test_df = test_df
        self.draw_original_training_data_figure(self.df, self.test_df)
        predict_df = df.copy()
        predict_dfs = np.split(predict_df, [len(predict_df.columns)-1], axis=1)
        predict_X_df = predict_dfs[0]
        predict_y_df = predict_dfs[1]
        predict_X = predict_X_df.values.tolist()
        predict_y = predict_y_df.values.reshape(-1,).tolist()
        predict_result = mlp.test(predict_X, predict_y)
        self.predict_result = predict_result
        self.data_num = len(predict_result["output_list"])
        # self.draw_predict_training_data_figure(self.predict_result, self.test_df)


root = tk.Tk()
app = Application(root)
root.mainloop()