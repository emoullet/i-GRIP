import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([0,1,2], [0,1,2])
# self.plotter =animation.FuncAnimation(plt.figure(), self.update_plot, interval=30)
plt.show()
xs = []
ys = []

# # This function is called periodically from FuncAnimation
# def animate(i, xs, ys):

#     # Add x and y to lists
#     xs.append(i)
#     ys.append(i ** 2)  # Replace this with your real-time data

#     # Limit x and y lists to 20 items
#     xs = xs[-20:]
#     ys = ys[-20:]

#     # Draw x and y lists
#     ax.clear()
#     ax.plot(xs, ys)

#     # Format plot
#     plt.xticks(rotation=45, ha='right')
#     plt.subplots_adjust(bottom=0.30)
#     plt.title('Real-time plot example')
#     plt.ylabel('Variable value')

# # Set up plot to call animate() function periodically
# ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
# plt.show()

# # import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation
# # import numpy as np
# # import threading
# # import random
# # import time
# # class MyDataClass():
# #     def __init__(self):
# #         self.XData = [0]
# #         self.YData = [0]
# # class MyPlotClass():
# #     def __init__(self, dataClass):
# #         self._dataClass = dataClass
#         self.hLine, = plt.plot(0, 0)
#         self.ani = FuncAnimation(plt.gcf(), self.run, interval = 1000, repeat=True)
#     def run(self, i):  
#         print("plotting data")
#         self.hLine.set_data(self._dataClass.XData, self._dataClass.YData)
#         self.hLine.axes.relim()
#         self.hLine.axes.autoscale_view()
# class MyDataFetchClass(threading.Thread):
#     def __init__(self, dataClass):
#         threading.Thread.__init__(self)
#         self._dataClass = dataClass
#         self._period = 0.25
#         self._nextCall = time.time()
#     def run(self):
#         while True:
#             print("updating data")
#             # add data to data class
#             self._dataClass.XData.append(self._dataClass.XData[-1] + 1)
#             self._dataClass.YData.append(random.randint(0, 256))
#             # sleep until next execution
#             self._nextCall = self._nextCall + self._period;
#             time.sleep(self._nextCall - time.time())
# data = MyDataClass()
# plotter = MyPlotClass(data)
# fetcher = MyDataFetchClass(data)
# fetcher.start()
# plt.show()
# #fetcher.join()