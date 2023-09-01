import matplotlib.pyplot as plt
ious = [0.8535253826773985, 0.8535253826773985, 0.8535253826773985, 0.8534867284894054, 0.8542592104878121,
        0.8510948167074027, 0.7629087650627254, 0.483047846838091, 0.06850023888412654, 0.011011978536004266,
        0.0]

noises = [0., 1e-7 , 1e-6 , 1e-5, 1e-4, 1e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 1]
# plt.plot(noises, ious)
plt.plot(noises, ious, color='blue', label='iou')
plt.xscale('log')
plt.yscale('log')

# Set labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Equal Tick Intervals (Logarithmic Scale)")
plt.grid()
# Display the plot
plt.show()
