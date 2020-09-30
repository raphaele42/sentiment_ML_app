fig, axs = plt.subplots(1, 2) 
        axs[0, 0].plot_precision_recall_curve(X_test, y_test) 
        axs[0, 1].plot_precision_recall_curve(X_test, y_test)
        #axs[0, 2].plot_precision_recall_curve(model, X_test, y_test)
  
        fig.suptitle('matplotlib.pyplot.subplots() Example') 
        plt.show()
        
        disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


arr = np.random.normal(1, 1, size=100)
>>> fig, ax = plt.subplots()
>>> ax.hist(arr, bins=20)
>>>
>>> st.pyplot(fig)


 # First create some toy data: 
#    x = np.linspace(0, 1.5 * np.pi, 100) 
#    y = np.sin(x**2)+np.cos(x**2) 
#  
#    fig, axs = plt.subplots(2, 2, 
#                        subplot_kw = dict(polar = True)) 
#    axs[0, 0].plot(x, y) 
#    axs[1, 1].scatter(x, y) 
#  
#    fig.suptitle('matplotlib.pyplot.subplots() Example') 
#    plt.show()