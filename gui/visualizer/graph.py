import  numpy   as  np


class Histogram(object):
    def initialize(self, array, figure, axes, margin=4):
        axes.clear()
        array = array[array > 0].ravel()
        freq, _, _ = axes.hist(array, 128, density=True, facecolor="gray") 

        self.xmin, self.xmax = array.min(), array.max()      

        self.lowerspan = axes.axvspan(
            self.xmin, self.xmin+margin, 
            facecolor='blue', 
            edgecolor='yellow', 
            alpha=.2
        )
        self.upperspan = axes.axvspan(
            self.xmax-margin, self.xmax, 
            facecolor='blue', 
            edgecolor='yellow', 
            alpha=.2
        )
 
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xlim([1, 254])
        axes.set_ylim([0, 1.1*np.max(freq)])
        
        figure.draw()

    def drawLocalHistogram(self, array, local, figure, axes, margin=4):
        axes.clear()
        array = array[array > 0].ravel()
        axes.hist(array, 128, density=True, facecolor="gray")
        
        freq, _, _ = axes.hist(
            local.ravel(), 128, density=True, facecolor="green"
        )

        self.xmin, self.xmax = array.min(), array.max()

        self.lowerspan = axes.axvspan(
            self.xmin, self.xmin+margin,
            facecolor='blue',
            edgecolor='yellow',
            alpha=.2
        )
        self.upperspan = axes.axvspan(
            self.xmax-margin, self.xmax,
            facecolor='blue',
            edgecolor='yellow',
            alpha=.2
        )

        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xlim([1, 254])
        axes.set_ylim([0, 1.1*np.max(freq)])

        figure.draw()

    def shiftRange(self, xposmin, xposmax, figure, axes, margin=4):
        lowerxy = [
            [self.xmin, 0],
            [self.xmin, 1],
            [xposmin+margin, 1],
            [xposmin+margin, 0],
            [self.xmin, 0]
        ]
        upperxy = [
            [xposmax-margin, 0],
            [xposmax-margin, 1],
            [self.xmax, 1],
            [self.xmax, 0],
            [xposmax-margin, 0]
        ]

        self.lowerspan.set_xy(lowerxy)
        self.upperspan.set_xy(upperxy)

        figure.draw()
