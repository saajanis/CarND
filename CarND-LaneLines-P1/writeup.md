# **Finding Lane Lines on the Road** 

Final video 1 [solidWhiteRight](https://youtu.be/A0AtQF_pikw)

Final video 2 [solidYellowLeft](https://youtu.be/gQJpdq8N0XY)

### The pipeline

---

**In this project, I used the provided helper function and tuned them to work on the provided images and then on the videos.<br />**

The function run_pipeline() has calls to all the helper functions. The components of the pipeline are applied in the following succession:
* <b>Grayscale:</b><br />
The image is first converted to grayscale. It works fine on these images/videos but I'm not sure if this is the best strategy for images with shade on the line such as [this image](http://imgur.com/a/xZ23d) in the challenge video.<br/>
Since we know what color of lines to expect (white/yellow etc.), we can apply a more informed color selection technique?<br /><br />
* <b>Gaussian Blur:</b><br />
I chose a kernel of size 15.<br /><br />
* <b>Canny Edge Detector:</b><br />
Used parameters low_threshold=0 and high_threshold=100.<br /><br />
* <b>Region of Interest:</b><br />
I selected a 4-sided polygon as my region of interest (see pictures below)<br /><br />
* <b>Hough Transform:</b><br />
I used min_line_len = 5 and max_line_gap = 25<br /><br />
* <b>Extrapolate line segments to two lines (left and right lane):</b><br />
    * I first divide the line segments into left and right lane line segments by checking their 'x' coordinate value to see if they fall in the left or right half of the frame.
    * For segments in each category, I use the np.polyfit() function to detect their average slope and intercept.
    * Since I know the y-coordinates of the region of interest for each half of the frame, I use that value in conjunction with the slope and intercept detected in the previous step to detect their corresponding x values.
    * Finally, for each section, I have the bounding x, y coordinates of the lane lines which I draw on the image.
<br /><br />
* <b>Draw the detected lines on the original image:</b><br />
This step does a weighted merge of the image with lines from the previous step with the original image<br /><br />

**The following are the images from each step in the pipeline:**<br />

* <b>Grayscale:</b><br />
<figure>
<img src="http://i.imgur.com/5lEBgF0.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/hdsSMtO.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/q4E8uD3.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/mAVZ6LL.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/rKZSVSo.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/YsbPdmX.jpg" width="380" alt="Grayscale Image" />
</figure><br /><br />
* <b>Gaussian Blur:</b><br />
<figure>
<img src="http://i.imgur.com/CD6YDQD.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/7NYzUa4.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/Gmlbj0Z.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/gCWTVme.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/TRAWkp2.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/6XsCygx.jpg" width="380" alt="Grayscale Image" />
</figure>
* <b>Canny Edge Detector:</b><br />
<figure>
<img src="http://i.imgur.com/p0gk6Or.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/pQW6DMC.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/xbDjo5t.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/2OU2kC4.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/HBq9GOf.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/ctqH8CU.jpg" width="380" alt="Grayscale Image" />
</figure>
* <b>Region of Interest:</b><br />
<figure>
<img src="http://i.imgur.com/gBB6hT3.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/YGTHS5s.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/3isaaSN.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/7vpbaqz.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/aPmOwFa.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/Hvjcj0w.jpg" width="380" alt="Grayscale Image" />
</figure>
* <b>Hough Transform:</b><br />
<figure>
<img src="http://i.imgur.com/CWZQqQY.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/wnwbbps.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/EIUeB7w.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/DvSlvwP.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/PmIXJat.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/EBBBcQs.jpg" width="380" alt="Grayscale Image" />
</figure>
* <b>Extrapolate line segments to two lines (left and right lane):</b><br />
<figure>
<img src="http://i.imgur.com/eYtgApe.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/Z9xnOh7.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/8fubLwy.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/E5Dqw1O.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/5KMYjB4.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/z3emEMt.jpg" width="380" alt="Grayscale Image" />
</figure>
* <b>Draw the detected lines on the original image:</b><br />
<figure>
<img src="http://i.imgur.com/B7HAOGr.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/2mpQzZ3.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/cIE5m4f.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/YK2lD5H.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/XiVvwxe.jpg" width="380" alt="Grayscale Image" />
<img src="http://i.imgur.com/kC3MtPn.jpg" width="380" alt="Grayscale Image" />
</figure>

---

### Potential shortcomings with the current pipeline

* If the lane lines are curved, fitting a curve of degree 1 can fail and this pipeline won't work.

* If all or part of the lane lines are not within the region of interest I've chosen, the pipeline won't see the lane lines.

### Possible improvements to the pipeline

* Moving to a higher order polynomial to fit the lines would help scale to more kinds of lane lines.
* Converting to GrayScale may be prone to errors since we lose out on important information about where the lanes are by distinguising them with their color (e.g. yellow). I would like to use the color characteristics of lane lines to filter them out before applying Canny and Hough filters.
* Hardcoding region of interest sounds like a bad idea. I would like to work on choosing it dynamically based on the current frame's composition.

### Future work

* I would like to see how the pipeline performs with an IR camera for when the car is driving at night and also how it can be made more robust to frames in which the camera is positioned against sunlight.