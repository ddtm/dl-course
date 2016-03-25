Seminar 6: Hacking R-CNN
====================

**WARNING: This seminar was not thoroughly tested so expect weird bugs. Please report if you found one!**

In this assignment, you will hack the existing code for object detection (https://github.com/rbgirshick/py-faster-rcnn) in order to make it usable with **Theano**-based models.

Originally, Rob Girshick uses [Caffe](http://caffe.berkeleyvision.org/) as a backend for deep learning. This is not very good for us as Caffe is substantially different from Theano, and one cannot simply replace one framework with the other. Luckily, the training and the testing procedures are built on top of the [Python interface](http://caffe.berkeleyvision.org/tutorial/interfaces.html) (instead of the Caffe's native CLI+protobufs combo) which makes it possible to locate and exterminate Caffe-contaminated parts and fill the gaps with appropriate wrappers around Theano machinery.

While it sounds like an easy task, this surgery still requires good familiarity with the RCNN's internals, so you will have to go through the Rob's slides and the code and make sure that you understand what is happening in each of the modules.

We (the TAs) will try to simplify you life by doing significant codebase reduction and adapation and uploading the scaffolding onto the course page, but you are welcome to dive deep into the original implementation and conduct clean-up and transplantaion completely on your own.

All your custom code should go into, well, `custom/` folder. There are two other files (`lib/utils/{train test}.py`) which should be modified. We suggest you to create separate classes for the **model** (holding a Theano expression which can be later compiled into proper function) and for the **solver** (conducting actual SGD steps for the model parameters). Look at the [original code](https://github.com/rbgirshick/py-faster-rcnn) for reference. Check [Lasagne recipes](https://github.com/Lasagne/Recipes) for pretrained networks.

The submission will be considered successful if:
  * The updated training and testing code runs with no errors (check `experiments/scripts/fast_rcnn.sh` script)
  * You have demonstrated that fine-tuning of the pre-trained network improves the result on PASCAL VOC2007

There is Makefile for compiling cython code (which can be found in `lib/` folder). Running make all should be sufficient.

The Rob's code makes use of a non-standard layer called **ROIPooling**. This trick gives a **very** nice speed-up but is absent from the Theano package. That's why we are not requiring you to incorporate it into your solutions. Feel free to go with a vanilla RCNN system (i.e. feed resized crops into the very beginning of your network).

It goes without saying, that you can also go ahead and wrap the original CUDA kernel into a Theano Op or write your own CPU implementation from scratch. In that case, you can expect immense respect for your engineering skills as well as generous bonus points.
