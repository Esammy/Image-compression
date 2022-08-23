from skimage import io
from sklearn.cluster import KMeans
from matplotlib.image import imread 
from skimage import measure, metrics
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os


class ImageCompression:
    def __init__(self, image, cluster=2):
        self.image = image
        self.cluster = cluster
        
        
    def save(self, compressed_image):
        path = str(self.cluster) + '.jpg'
        io.imsave(path, compressed_image)
        #print('Saved succesfully')
    
    def kmeans_compression(self):
        image = imread(self.image)
        cluster = self.cluster
        original = image.copy()
          
        print("for K = ", cluster, "\n")
        #Dimension of the original image
        rows = image.shape[0]
        cols = image.shape[1]
        
        #Flatten the image
        image = image.reshape(rows*cols, 3)
        
        #Implement k-means clustering to form k clusters 
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(image)
          
        #replace each pixel value with its neaaby cetroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
        
        #reshape the image to original dimension
        compressed_image = compressed_image.reshape(rows, cols, 3)
        self.save(compressed_image)
        return compressed_image
    
    
        
    def display(self, compressed_image):
        #image_compressed = imread(path, 3)
        plt.axis('off')
        plt.title('cluster' + str(self.cluster) + 'image')
        plt.imshow(compressed_image)
        
    def mse(self, compressed_image):
        original = imread(self.image)
        # Calculate the mean square error of the original and the predicted
        mse = metrics.mean_squared_error(original, compressed_image)
        #print('mean square error is: ', mse)
        return mse
    
    def convert_bytes(self, num):
        """
        this function will convert bytes to MB.... GB... etc
        """
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if num < 1024.0:
                return "%3.1f %s" % (num, x)
            num /= 1024.0
            
    def file_size(self):
        """
        this function will return the file size
        """
        path = str(self.cluster) + '.jpg'
        if os.path.isfile(path):
            file_info = os.stat(path)
            return self.convert_bytes(file_info.st_size)
    
    def compression_ratio(self, compressed_image):
        original = imread(self.image)
        path = path = str(self.cluster) + '.jpg'
        return (original.shape[0]*original.shape[1]) / os.stat(path).st_size
        

if __name__ == "__main__":
    image = '4.jpg'
    test = ImageCompression(image)
    
    #from Kmeans import ImageCompression
    img = '4.jpg'
    test = ImageCompression(img, cluster=1)
    im = test.kmeans_compression()
    test.display(im)
