#$LAN=PYTHON$
#M133040059 李旭智

import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import tkinter.filedialog as filedialog
import math
import copy
from fractions import Fraction

infinite=1e6
canvas_click=False
paused = True 
running=False
def stop():
    
    global paused
    while paused:
        root.update()  # 不斷刷新 Tkinter 界面，保持 GUI 響應
    if(running):return
    paused = True  # 重置 paused 狀態，等待下一次暫停

def next_step():
    global paused
    paused = False

def run():
    global running
    global paused
    if not paused:
        return
    paused=False
    running=True


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Voronoi diagram")
        
        # 初始化變數
        self.points = []
        self.current_step = 0
        self.is_mouse_input = True
        self.test_cases = deque()  # 存儲所有測資組
        
        self.original_edges=[]
        self.canvas_edges=[]
        
        # 創建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 創建畫布
        self.canvas = tk.Canvas(self.main_frame, width=600, height=600, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2)
        
        # 創建按鈕框架
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # 創建按鈕
        self.testcase_button = ttk.Button(self.button_frame, text="Next testcase", command=self.next_animation)
        self.testcase_button.grid(row=0, column=0, padx=5)
        
        self.run_button = ttk.Button(self.button_frame, text="Run", command=run)
        self.run_button.grid(row=0, column=1, padx=5)

        self.step_button = ttk.Button(self.button_frame, text="Step by Step",command=next_step)
        self.step_button.grid(row=0, column=2, padx=5)
        
        
        
        
        
        # 創建清除按鈕
        self.clear_button = ttk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=0, column=3, padx=5)
        

        
         # 顯示剩餘測資數量的標籤
        self.remaining_cases_label = ttk.Label(self.button_frame, text="Remaining cases: 0")
        self.remaining_cases_label.grid(row=0, column=4, padx=5)

        
        
        
        
        # Textbox for output
        self.output_frame = ttk.LabelFrame(self.main_frame,text="Output:")
        self.output_frame.grid(row=0, column=2, padx=10, sticky=(tk.W,tk.N, tk.S))
        #設定滾動條
        self.scrollbar = ttk.Scrollbar(self.output_frame, orient='vertical')
        self.scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.output_text = tk.Text(self.output_frame, width=25, height=15, wrap='word', font=("Arial", 10),yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.output_text.yview)    # 設定 scrollbar 綁定 text 的 yview
        self.output_text.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Add Button for Loading File
        self.load_button = ttk.Button(self.output_frame, width=25,text="Load Points from File", command=self.load_points_from_file)
        self.load_button.grid(row=4, column=0, padx=5)
        
        #output file button
        self.outputFile_button = ttk.Button(self.output_frame,width=25, text="Output File", command=self.output_points_and_edges_file)
        self.outputFile_button.grid(row=2, column=0, padx=5)
        
        # Add Button for Displaying Points and Edges
        self.displayFile_button = ttk.Button(self.output_frame, width=25, text="Display File", command=self.display_diagram_from_file)
        self.displayFile_button.grid(row=3, column=0, padx=5, pady=5)

        # 綁定滑鼠點擊事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
    def on_canvas_click(self, event):
        global canvas_click
        canvas_click=True
        if not self.is_mouse_input:
            return
        
        x, y = event.x, event.y
        print(x,y)
        if 0 <= x <= 600 and 0 <= y <= 600:
            self.points.append([x, y])
            p=[x,y]
            self.draw_point(p)
        

    def draw_point(self, point, color="black"):
        x=point[0]
        y=point[1]
        point_size = 3
        self.canvas.create_oval(x-point_size, y-point_size, 
                              x+point_size, y+point_size, 
                              fill=color, outline=color)
                              
    def draw_line(self, line, color="blue"):
        x1=line[0][0]
        y1=line[0][1]
        x2=line[1][0]
        y2=line[1][1]
        self.canvas.create_line(x1, y1, x2, y2, fill=color)
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.points = []
        self.original_edges=[]
        self.canvas_edges=[]
        
        
    def next_animation(self):
        
        
        print("===================================================")
        global canvas_click
        if(canvas_click):
            self.test_cases.appendleft(self.points)
        
        canvas_click=False
        # 如果有測資，則執行下一組
        if self.test_cases :
            # Clear previous output
            app.output_text.delete(1.0, tk.END)
            app.clear_canvas()
            self.points = self.test_cases.popleft()

            
            #創建diagram object
            diag=Diagram(self.points)

            # 畫出所有點
            for point in diag.points:
                self.draw_point(point)
            
            stop()
            diag=diag.divide()
            

            # 更新剩餘測資數量顯示
            self.remaining_cases_label.config(text=f"Remaining cases: {len(self.test_cases)}")
            
            print("diag.ori edges:",diag.original_edges)

            diag.draw_diagram("green")
            
            c_edges=[]
            for edge in diag.original_edges:
                start=[edge[0][0],edge[0][1]]
                end=[edge[1][0],edge[1][1]]
                c_edge=[start,end]
                c_edges.append(c_edge)
            #print("c edges:",c_edges)
            diag.canvas_edges=c_edges
            #print("canvas edge",diag.canvas_edges)
            diag.sort_canvas_edges()
            diag.turn_canvas_into_output_Edge()
            print("canvas edge",diag.canvas_edges)


            diag.display_points_and_edges()
            app.points=diag.points
            app.canvas_edges=diag.canvas_edges
            
        #用滑鼠選點
        # elif self.points and not self.canvas_edges:
            
        #     app.output_text.delete(1.0, tk.END)
            
        #     #create diagram object
        #     diag=Diagram(self.points)

        #     # 畫出所有點
        #     for point in diag.points:
        #         self.draw_point(point)
        #     stop()
        #     diag=diag.divide()
            
        #     diag.draw_diagram("green")

        #     diag.canvas_edges=copy.deepcopy(diag.original_edges)
            

        #     diag.sort_canvas_edges()
        #     diag.turn_canvas_into_output_Edge()

        #     diag.display_points_and_edges()

        #     app.points=diag.points
        #     app.canvas_edges=diag.canvas_edges

        else:
            messagebox.showwarning("Warning", "No more test cases or insufficient points!")
            
        global running 
        running=False
        

 
            
     
    #從檔案輸入       
 
    def load_points_from_file(self):

        self.test_cases.clear()
        # Load a text file containing points and edges
        file_path = filedialog.askopenfilename(title="Select Point Data File", filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return
        

        with open(file_path, 'r',encoding='utf-8') as file:
            self.points = []  # Reset points list
            for line in file:
                if line.startswith('#') or line.startswith('\n'):
                    continue
                if line.startswith('0'):
                    return
                
                n = int(line)
                current_points = []
                for _ in range(n):
                    point_line = next(file).strip()
                    # 跳过空行和注释
                    while not point_line or point_line.startswith("#"):
                        point_line = next(file).strip()
                    
                    # 解析点的 x 和 y 坐标
                    x, y = map(int, point_line.split())
                    if [x,y] not in current_points:
                        current_points.append([x, y])
                
                # 将这组测试数据加入到 test_cases 列表中
                self.test_cases.append(current_points)
        self.remaining_cases_label.config(text=f"Remaining cases: {len(self.test_cases)}")
        messagebox.showinfo("Load Points", "Points and edges loaded successfully from file.")





    def display_diagram_from_file(self):
        file_path = filedialog.askopenfilename(title="Select Points and Edges File", filetypes=[("Text Files", "*.txt")])
        
        # 檢查 file_path 是否有效
        if not file_path:
            print("No file selected.")
            return

        with open(file_path, 'r',encoding='utf-8') as file:
            self.points = []  # Reset points list
            self.canvas_edges=[]
            self.original_edges=[]
            for line in file:
                if line.startswith('#') or line.startswith('\n'):
                    continue
                if line.startswith('0'):
                    return
                if line.startswith("P"):
                    current_point=[]
                    _, x, y = line.split()  # 分割後取 x 和 y 值
                    x, y = int(x), int(y)   # 轉換為整數
                    current_point=[x,y]  # 添加到 current_points
                    self.points.append(current_point)
                    app.draw_point(current_point,"green")
                    
                # 判斷是否為邊 E 開頭
                elif line.startswith("E"):
                    _, x1, y1, x2, y2 = line.split()  # 分割並取 x1, y1, x2, y2
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 轉換為整數
                    # 可以在這裡處理邊的數據，例如畫邊或儲存
                    edge=[[x1,y1],[x2,y2]]   
                    self.canvas_edges.append(edge)
                    app.draw_line(edge,"green")
                    
                     
            

    #輸出點和邊的文字檔
    def output_points_and_edges_file(self):
        with open('output.txt', 'w', encoding='utf-8') as f:
            for point in self.points:
                f.write(f"P {point[0]} {point[1]}\n")
            for edge in self.canvas_edges:
                f.write(f"E {edge[0][0]} {edge[0][1]} {edge[1][0]} {edge[1][1]}\n")
    


class Diagram:
    def __init__(self,points):
        self.points=points
        self.original_edges=[]  #edge=[[start point],[endpoint],[parent point1],[parent point2]]
        self.canvas_edges=[]
        
        #輸出點和邊
    def display_points_and_edges(self):
        
        
        # Display Points
        for i, (x, y) in enumerate(self.points, start=1):
            app.output_text.insert(tk.END, f"P {x} {y}\n")
        
        #print("canvas edge:",self.canvas_edges)
        # Display Edges
        for edge in self.canvas_edges:
            app.output_text.insert(tk.END, f"E {edge[0][0]} {edge[0][1]} {edge[1][0]} {edge[1][1]}\n")

        #print("last canvas_edge:",self.canvas_edges)


    def draw_diagram(self,color):
        print("draw diagram")
        
        for point in self.points:
            app.draw_point(point,color)

        for edge in self.original_edges:
            app.draw_line(edge,color)

    def drawAllPoints(self,points,color):
        for point in points:
            app.draw_point(point,color)
    def drawAllEdges(self,edges,color):
        for edge in edges:
            app.draw_line(edge,color)

    def are_points_collinear(self):
    
    # 判断点列表中的所有点是否共线。
    
   
        if len(self.points) < 3:
            # 少于3个点总是共线
            return True

        # 取第一个点和第二个点计算初始斜率
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        
        for i in range(2, len(self.points)):
            x3, y3 = self.points[i]
            # 判断三点是否共线，使用向量叉积法
            if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
                return False
        
        return True
            

    def draw_convexhull(self,convexhull, color):
 
        if len(convexhull) < 2:
            
            return

        # 依次连接凸包点
        for i in range(len(convexhull)):
            x1, y1 = convexhull[i]
            x2, y2 = convexhull[(i + 1) % len(convexhull)]  # 闭合到首点
            line=[[x1,y1],[x2,y2]]
            app.draw_line(line,color)
#======================================算外心


    
    # 计算质心
    def calculate_centroid(self,points):
        x_sum = sum([p[0] for p in points])
        y_sum = sum([p[1] for p in points])
        return (x_sum / len(points), y_sum / len(points))

    # 计算点相对于质心的角度
    def angle_from_centroid(self,point, centroid):
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]
        return math.atan2(dy, dx)

    # 按逆时针顺序排序
    def sort_points_counterclockwise(self,points):
        centroid = self.calculate_centroid(points)
        # 计算每个点的角度并排序
        sorted_points = sorted(points, key=lambda p: self.angle_from_centroid(p, centroid), reverse=True)
        return sorted_points

    def circumcenter(self,A, B, C):

        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))


        Ux = ( (A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1]) ) / D


        Uy = ( (A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0]) ) / D

        #print("circumcenter:"+str(Ux)+" "+str(Uy)+"\n")
        return (Ux, Uy)

#============================畫出三點以下的圖 
    #畫出外心和三點的中垂線
    def calculate_centroid_and_normals(self,points):
    
        # 按逆時針排序點
        points = self.sort_points_counterclockwise(points)
        
        # 計算3個點的外心坐標
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        
        
        
        centroid = self.circumcenter(points[0],points[1],points[2])
        
        # 計算3個邊的法向量
        normal1 = [-(y2 - y1), x2 - x1,points[0],points[1]]
        normal2 = [-(y3 - y2), x3 - x2,points[1],points[2]]
        normal3 = [-(y1 - y3), x1 - x3,points[0],points[2]]
        
        normals = [normal1, normal2, normal3]

        return centroid, normals
    
    def cal_voronoi(self):
        stop()
        if len(self.points) < 3:
            self.cal_perpendicular_bisectors()
        else:
            # 計算向量a和b
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            x3, y3 = self.points[2]
            ax = x2 - x1
            ay = y2 - y1
            bx = x3 - x2
            by = y3 - y2

            cross_product=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)
            # 判斷是否共線
            if cross_product == 0:
                self.cal_perpendicular_bisectors()
                return 

            else:
                centroid, normals = self.calculate_centroid_and_normals(self.points)
            
            # 在外心處畫一個點
            # self.draw_point(centroid, "red")
            
            # 沿著3個法向量繪製射線
            
            
            for normal in normals:
                start=[centroid[0], centroid[1]]
                end=[centroid[0] + infinite * normal[0], centroid[1] + infinite * normal[1]]
                edge=[start,end,normal[2],normal[3]]
                



                self.original_edges.append(edge)

                

    def add_two_points_perpendicular_in_merge(self,a,b):
        
        x1, y1 = a
        x2, y2 = b
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        dx=x2-x1
        dy=y2-y1        
        
        ux=-dy
        uy=dx
        start=[mid_x-infinite*ux,mid_y-uy*infinite]
        end=[mid_x+infinite*ux,mid_y+uy*infinite]
        edge=[start,end,a,b]
        self.original_edges.append(edge)
        app.draw_line(edge,"purple")
        

    def cal_perpendicular_bisectors(self):

        if len(self.points)==1:return

        elif len(self.points)==2:
            print("2點算中垂")
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            dx=x2-x1
            dy=y2-y1        
            
            ux=-dy
            uy=dx
            start=[mid_x-infinite*ux,mid_y-uy*infinite]
            end=[mid_x+infinite*ux,mid_y+uy*infinite]
            edge=[start,end,self.points[0],self.points[1]]
            self.original_edges.append(edge)
            
            print("mid line:",edge)
            


        elif len(self.points)==3:
            print("3點算中垂")
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            x3, y3 = self.points[2]

            # 计算两条中垂线的端点坐标
            mid_x1 = (x1 + x2) / 2.0
            mid_y1 = (y1 + y2) / 2.0
            mid_x2 = (x2 + x3) / 2.0
            mid_y2 = (y2 + y3) / 2.0
            dx=x2-x1
            dy=y2-y1        
            # 计算两条中垂线的斜率
            ux=-dy
            uy=dx
            
            start1=[mid_x1-infinite*ux,mid_y1-uy*infinite]
            end1=[mid_x1+infinite*ux,mid_y1+uy*infinite]
            edge1=[start1,end1,self.points[0],self.points[1]]
            
            
            start2=[mid_x2-infinite*ux,mid_y2-uy*infinite]
            end2=[mid_x2+infinite*ux,mid_y2+uy*infinite]
            edge2=[start2,end2,self.points[1],self.points[2]]

            self.original_edges.append(edge1)
            self.original_edges.append(edge2)
            
 

    def turn_canvas_into_output_Edge(self):
        print("turn output")
        for i in range(len(self.canvas_edges)):
            edge=self.canvas_edges[i]
        
            x1=edge[0][0]
            y1=edge[0][1]
            x2=edge[1][0]
            y2=edge[1][1]

            newedge = []

            
            dx=x2-x1
            dy=y2-y1
        

            if x1<0 or x1>600:
                if x1<0:
                    
                    y1=y1+dy/dx*(-x1)
                    x1=0
                elif x1>600:
                    y1=y1+dy/dx*(600-x1)
                    x1=600
            
            

            if y1<0 or y1<600:
                if y1<0.0:
                    x1=dx/dy*(-y1)+x1
                    y1=0.0
                elif y1>600:
                    x1=x1+dy/dx*(600-y1)
                    y1=600
            x1=round(x1)
            y1=round(y1)
            start=[x1,y1]

            if x2<0 or x2>600:
                if x2<0:
                    
                    y2=y2+dy/dx*(-x2)
                    x2=0.0
                elif x2>600:
                    m=dy/dx
                    
                    y2=y2+dy/dx*(600-x2)
                    x2=600
            
            

            if y2 <0 or y2 >600:
                if y2<0:
                    
                    dx=x2-x1
                    dy=y2-y1
                    x2=dx/dy*(-y2)+x2
                    y2=0
                
                    
                elif y2>600:
                    
                    dx=x2-x1
                    dy=y2-y1
                    x2=x2+dx/dy*(600-y2)
                    y2=600
                    
            
            x2=round(x2)
            y2=round(y2)

            end=[x2,y2]
            newedge=[start,end]
            #print("edge",edge,"\nnew edge",newedge)
            self.canvas_edges[i]=newedge
            #print("edge",edge)

    def sort_points(self):
        #排序所有點
        self.points=sorted(self.points,key=lambda p:(p[0],p[1]))

    def sort_canvas_edges(self):
        #排序邊
        for i,edge in enumerate(self.canvas_edges):
            self.canvas_edges[i]=sorted(edge,key=lambda p:(p[0],p[1]))
        
        self.canvas_edges=sorted(self.canvas_edges,key=lambda e:(e[0][0],e[0][1],e[1][0],e[1][1]))



#==================================convexhull
    def pointDist(self,a,b):
        x1=a[0]
        y1=a[1]
        x2=b[0]
        y2=b[1]
        dist=math.hypot(x2-x1,y2-y1)
        return dist

    def getMinYpoint(self):
        miny=math.inf
        minx=math.inf
        minPoint=None
        for point in self.points:
            if miny>point[1]:
                minx=point[0]
                miny=point[1]
                minPoint=point
            elif miny==point[1]:
                if minx > point[0]:
                    minPoint=point
                else:
                    continue
        return minPoint

    def ccw(self,a,b,c):
        area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

        if area<0:return -1 #clockwise
        if area>0: return 1 #conterclockwise

        return 0 #collinear

  
      
    def convexhull(self):
        #print("convexhull function")
        hull=[]
        startPoint=self.getMinYpoint()

        #print("minY_point:",startPoint)

        hull.append(startPoint)
        prevertex=startPoint
        prepre_point=None
        
        while(True):
            
            candidate=None
            #print("points\n",self.points)
            for point in self.points:
                if point==prevertex: continue
                if candidate==None:
                    candidate=point
                    
                    continue
                
                ccw=self.ccw(prevertex,candidate,point)
                if ccw==0 and self.pointDist(prevertex,candidate)>self.pointDist(prevertex,point) :
                    if point==prepre_point:break
                    #print("ccw=0","prevertex:",prevertex,"candidate:",candidate,"point",point,"prepre point",prepre_point)
                    candidate=point
                elif ccw<0:
                    
                    candidate=point
                    #print("ccw<0","prevertex:",prevertex,"candidate:",candidate,"point",point)

            #print("prevertex:",prevertex,"candidate:",candidate)
            if candidate==startPoint:break
            hull.append(copy.deepcopy(candidate))
            #print("hull append:",candidate,"\n")
            prepre_point=prevertex
            prevertex = candidate

        return hull

        

#==================================hyperplain
    def cut_subdiagram_line(self,main_line,cut_line,intersection,left_diagram:'Diagram',right_diagram:'Diagram',index):
        
        print("cut sub diagram line function")
        
        print("main line:",main_line,"\ncut line",cut_line,"\nintersection",intersection)
        

        #判斷cut line屬於左右圖 再判斷要順還逆的segment 左圖留順 右圖留逆
        if cut_line in left_diagram.original_edges:

            #cut line in left
            #print("cut line in left")
            
            

            ccw=self.ccw(main_line[0],main_line[1],cut_line[0])
            print("cut left ccw=",ccw)
            if ccw>0:#cut line 起點逆時針 代表在右邊 起點改成intersection
                
                self.original_edges[index][0]=intersection
            elif ccw<0: #cut line 起點順時針 代表在左邊 終點改成intersection
                
                self.original_edges[index][1]=intersection
            
        else:
            #cut line in right
            #print("cut line in right")
            
            ccw=self.ccw(main_line[0],main_line[1],cut_line[0])
            print("cut right ccw=",ccw)
            if ccw>0: #cut line 起點逆時針 代表在右邊 終點改成intersection
                
                self.original_edges[index][1]=intersection
            elif ccw<0: #cut line 起點順時針 代表在左邊 起點改成intersection
                
                self.original_edges[index][0]=intersection
            

        print("cut line finish",self.original_edges[index])
        return copy.deepcopy(self.original_edges[index])


    

    def closest_intersection_and_line (self,main_line,lines_list,pre_top_line):
        print("closest intersection and line\n")
        start=main_line[0]
        min=math.inf
        top_point=[]
        top_line=[]
        intersection=None
        topline_index=0
        for i in range(len(lines_list)):
            
            
            print("mainline",main_line,"line,",lines_list[i])
            intersection=self.find_intersection(main_line,lines_list[i])
            print("intersection",intersection)
            if intersection is not None:
                d=self.pointDist(start,intersection)
                if lines_list[i]==pre_top_line:
                    continue
                elif min>d:
                    #print("min>d")
                    min=d
                    top_point=intersection
                    top_line=lines_list[i]
                    topline_index=i
                    
            else:continue
        
        print("top(p,e,i):",top_point,top_line,topline_index)
        return top_point,top_line,topline_index
        

    #找線段交點
    def find_intersection(self,edge1,edge2):
        #print("find intersection")
        
        # 提取線段端點
        x1, y1 = edge1[0]
        x2, y2 = edge1[1]
        x3, y3 = edge2[0]
        x4, y4 = edge2[1]
        
        # 計算行列式 det
        det = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        if det == 0:
            # 線段平行或重疊
            return None
        


        # 求解參數 t 和 u
        t = Fraction(((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / det)
        u = Fraction(((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / det)
        
        # 判斷是否在線段範圍內
        if 0 <= t <= 1 and 0 <= u <= 1:
            # 計算交點座標
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            return [x, y]
        
        # 不相交
        return None


    def cal_hyperplane(self,tang_lines,leftDiagram,RightDiagram):
        
        #print("hyperplane\n")
        low_tang_line_points=set()
        low_tang_line_points.add(tuple(tang_lines[1][0]))
        low_tang_line_points.add(tuple(tang_lines[1][1]))
        

        up_tang_diag=Diagram(tang_lines[0])
        up_tang_diag.cal_perpendicular_bisectors()
        now_bisector_line=up_tang_diag.original_edges[0]

        now_bisector_line_parents=set()
        now_bisector_line_parents.add(tuple(now_bisector_line[2]))
        now_bisector_line_parents.add(tuple(now_bisector_line[3]))
        #一開始所有邊都需找交點
        #new_lines=copy.deepcopy(self.original_edges)
        #print("self ori edges",self.original_edges)
        merge_edges=[]
        top_line=[]
        pre_bisector=[]
        
        while(now_bisector_line_parents!=low_tang_line_points):
            
            
            end_point,top_line,topline_index=self.closest_intersection_and_line(now_bisector_line,self.original_edges,top_line)
            
            
            now_bisector_line[1]=copy.deepcopy(end_point)
            print("now bisector:",now_bisector_line)
            
            #print("pre bisector",pre_bisector)

            if now_bisector_line[0]==now_bisector_line[1]:#如果now bisector起點和終點是同一個點 用前一個bisector來cut
                top_line=self.cut_subdiagram_line(pre_bisector,top_line,end_point,leftDiagram,RightDiagram,topline_index)
            else:
                top_line=self.cut_subdiagram_line(now_bisector_line,top_line,end_point,leftDiagram,RightDiagram,topline_index)
            
            #print("new top line:",top_line)
            
            #new_lines[topline_index]=copy.deepcopy(top_line)
            #print("ori edges:",self.original_edges)
            #print("new lines:",new_lines)
            print("now bisector line",now_bisector_line)
            print("top line",top_line)
            if now_bisector_line[0]!=now_bisector_line[1]:
                print("add merge edge:",now_bisector_line)
                merge_edges.append(copy.deepcopy(now_bisector_line))
                print("new merge:",merge_edges)
                app.draw_line(now_bisector_line,"purple")
                stop()
            
            #找出top line另一邊的parent
            if now_bisector_line[2]==top_line[2]:
                now_bisector_line[2]=copy.deepcopy(top_line[3])
            elif now_bisector_line[2]==top_line[3]:
                now_bisector_line[2]=copy.deepcopy(top_line[2])
            elif now_bisector_line[3]==top_line[2]:
                now_bisector_line[3]=copy.deepcopy(top_line[3])
            elif now_bisector_line[3]==top_line[3]:
                now_bisector_line[3]=copy.deepcopy(top_line[2])
            #用now bisector的new parent算新的now bisector
            new_diag=Diagram([now_bisector_line[2],now_bisector_line[3]])
            new_diag.cal_perpendicular_bisectors()
            #print("new diag edges:",new_diag.original_edges)

            pre_bisector=now_bisector_line.copy()
            now_bisector_line=copy.deepcopy(new_diag.original_edges[0])
            #把now bisector起點改成上一個交點
            now_bisector_line[0]=copy.deepcopy(end_point)
            #print("now bisector line:",now_bisector_line)
            now_bisector_line_parents=set()
            now_bisector_line_parents.add(tuple(now_bisector_line[2]))
            now_bisector_line_parents.add(tuple(now_bisector_line[3]))
        
        
        if now_bisector_line[0]!=now_bisector_line[1]:
            merge_edges.append(copy.deepcopy(now_bisector_line))
            print("add last hyper edge:",now_bisector_line)
            app.draw_line(now_bisector_line,"purple")
            
            
        print("merge line:",merge_edges)
        # print("ori_edge_in_hyper1:",self.original_edges)
        
        # print("ori_edge_in_hyper2:",self.original_edges)


        return merge_edges

    

#==================================divide and conquer
    
    def get_line_high_point(self,edge):
        if edge[0][1]==edge[1][1]:
            return edge[0]
        elif edge[0][1]>edge[1][1]:
            return edge[0]
        elif edge[0][1]<edge[1][1]:
            return edge[1]
        else :
            print("get line high point error")
            return None
    def get_line_low_point(self,edge):
        if edge[0][1]==edge[1][1]:
            return edge[0]
        elif edge[0][1]>edge[1][1]:
            return edge[1]
        elif edge[0][1]<edge[1][1]:
            return edge[0]
        else :
            print("get line low point error")
            return None

    def delete_useless_line(self,left_daig:'Diagram',right_diag:'Diagram',hyper_lines):
        print("delete useless line function")
        for i in range((len(self.original_edges)-1),-1,-1):
            side=''
            line = self.original_edges[i]
            if line in left_daig.original_edges:side='l'
            if line in right_diag.original_edges:side='r'
            print("line,",line,"\nside in ",side)
            is_out = False
            for h_line in hyper_lines:
                line_lowpoint=self.get_line_low_point(line)
                line_highpoint=self.get_line_high_point(line)
                hline_lowpoint=self.get_line_low_point(h_line)
                hline_highpoint=self.get_line_high_point(h_line)

                if (line_lowpoint[1] > hline_highpoint[1] or line_highpoint[1] < hline_lowpoint[1] \
                    or (line_lowpoint[1] < hline_lowpoint[1] and line_highpoint[1] > hline_highpoint[1])):
                    continue
                #voronoi line在目前這條hyperline的y值範圍內
                elif (line_lowpoint[1] > hline_lowpoint[1] and line_highpoint[1] < hline_highpoint[1]):
                    
                    if (self.ccw(hline_lowpoint, hline_highpoint, line_highpoint) > 0 and side == 'r'):
                    
                        is_out= True
                        break
                    
                    if (self.ccw(hline_lowpoint, hline_highpoint, line_highpoint) < 0 and side == 'l'):
                        is_out = True
                        break
                        
                    
                    #voronoi line只有一點(y比較低的點)在目前這條hyperline的y值範圍內
                elif (line_lowpoint[1] > hline_lowpoint[1] and line_highpoint[1] > hline_highpoint[1]):
                    
                    if (self.ccw(hline_lowpoint, hline_highpoint, line_lowpoint) > 0 and side == 'r'):
                        is_out = True
                        break
                        
                    if (self.ccw(hline_lowpoint, hline_highpoint, line_lowpoint) < 0 and side == 'l'):
                        is_out=True
                        break
                    
                #voronoi line只有一點(y比較高的點)在目前這條hyperline的y值範圍內
                elif (line_lowpoint[1] < hline_lowpoint[1] and line_highpoint[1] < hline_highpoint[1]):
                
                    if (self.ccw(hline_lowpoint, hline_highpoint, line_highpoint) > 0 and side == 'r'):
                    
                        is_out=True
                        break
                    if (self.ccw(hline_lowpoint, hline_highpoint, line_highpoint) < 0 and side == 'l'):
                    
                        is_out = True
                        break
            if is_out:
                print("del:",self.original_edges[i])
                del(self.original_edges[i])
                    
                    
                    




    def divide(self):
        print("divide function")
        

        self.sort_points()
        if len(self.points)>3:
            print("divide")
            subDiagram_left=Diagram(self.points[:len(self.points)//2])
            subDiagram_right=Diagram(self.points[len(self.points)//2:])

            print("left points",subDiagram_left.points)
            print("right points",subDiagram_right.points)           

            subDiagram_left=subDiagram_left.divide()
            
            
            #subDiagram_left.draw_diagram("blue")

            subDiagram_right=subDiagram_right.divide()
            print("left edges",subDiagram_left.original_edges)
            print("right edges",subDiagram_right.original_edges)

            #subDiagram_right.draw_diagram("red")



            return self.merge(subDiagram_left,subDiagram_right)
        else:
            print("3點以下 直接算圖")
            self.cal_voronoi()
            print("edges",self.original_edges)
            self.draw_diagram("grey")
            
            return self

    

    def merge(self,diagram_left:'Diagram',diagram_right:'Diagram') -> 'Diagram':
        stop()
        
        print("\n\n\nmerge================================\n")
        diagram_left.draw_diagram("blue")
        
        print("blue\n")
        print("blue points",diagram_left.points)
        print("blue edges",diagram_left.original_edges)
        diagram_right.draw_diagram("red")
       
        print("red\n")
        print("red points",diagram_right.points)
        print("red edges",diagram_right.original_edges)
        stop()
        all_points=diagram_left.points + diagram_right.points

        
        merged_diag=Diagram(all_points)
        if not merged_diag.are_points_collinear():
            #畫左右圖convex hull
            left_convexhull=diagram_left.convexhull()
            self.draw_convexhull(left_convexhull,"lightblue")
            stop()
            right_convexhull=diagram_right.convexhull()
            self.draw_convexhull(right_convexhull,"pink")
            stop()


            merged_convexhull=merged_diag.convexhull()
            #print("convexhull:\n",merged_convexhull)
            self.draw_convexhull(merged_convexhull,"orange")
            stop()
            

            #找公切線
            pre_point_in_merged_convexhull=merged_convexhull[0]
            merged_convexhull.append(copy.deepcopy(pre_point_in_merged_convexhull))

            tang_lines=[[],[]] # 0:上切線 1:下切線
            #print("left points:",diagram_left.points,"right points:",diagram_right.points)
            for now_point_in_merged_convexhull in merged_convexhull[1:]:

                #判斷兩點是否同邊來找公切線
                if (pre_point_in_merged_convexhull in diagram_right.points) and (now_point_in_merged_convexhull in diagram_left.points) :
                    
                    start=pre_point_in_merged_convexhull
                    end=now_point_in_merged_convexhull
                    tang_lines[0]=[start,end] #上切線
                elif (pre_point_in_merged_convexhull in diagram_left.points) and (now_point_in_merged_convexhull in diagram_right.points) :

                    start=pre_point_in_merged_convexhull
                    end=now_point_in_merged_convexhull
                    tang_lines[1]=[start,end] #下切線
                
                pre_point_in_merged_convexhull=now_point_in_merged_convexhull
            

            print("tang lines:",tang_lines)

            #找hyperplain 線
            merged_diag.original_edges=diagram_left.original_edges+diagram_right.original_edges
            print("old original_edges:",merged_diag.original_edges)
            hyper_lines=merged_diag.cal_hyperplane(tang_lines,diagram_left,diagram_right)
            print("hyper_lines",hyper_lines)

            
            print("new merged_diag.original_edges:\n",merged_diag.original_edges)
            stop()
            merged_diag.delete_useless_line(diagram_left,diagram_right,hyper_lines)
            
            merged_diag.original_edges.extend(hyper_lines)
            

        else:
            #畫左右圖convex hull
            left_convexhull=diagram_left.points
            self.draw_convexhull(left_convexhull,"lightblue")
            stop()
            right_convexhull=diagram_right.points
            self.draw_convexhull(right_convexhull,"pink")
            stop()
            merged_diag.original_edges=copy.deepcopy(diagram_left.original_edges)+copy.deepcopy(diagram_right.original_edges)
            
            self.draw_convexhull(merged_diag.points,"orange")
            stop()

            merged_diag.add_two_points_perpendicular_in_merge(diagram_left.points[-1],diagram_right.points[0])
            stop()
            
            

        print("merge diagram points:\n",merged_diag.points)
        print("merge diagram edges:\n",merged_diag.original_edges)
        app.clear_canvas()
        merged_diag.draw_diagram("black")
        stop()
        app.clear_canvas()
        
        return merged_diag
    


root = tk.Tk()
app = App(root)
root.mainloop()


    