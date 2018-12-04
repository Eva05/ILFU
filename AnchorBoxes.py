from __future__ import division
import numpy as np

from bounding_box_utils.bounding_box_utils import convert_coordinates

# Input Arguments we need , otherwise the behavior of the model is undefined :
    # - img_height int | IMAGE Height
    # - img_width int | IMAGE Width
    # - this_scale float | Scaling factor.Indicate the size of Anchor Boxes | float in [0,1]
    # - next_scale float | Next scaling factor.Indicate also the size of Anchor Boxes | Use only if {two_boxes_for_ar1}==True
    # - aspect_ratios (list,optional) | list of aspect for which default Anchor Boxes are generated
    # - two_boxes_for_ar1 (boolean) | Use only , when {aspect_ratios} contains the value 1
    # - this_steps (list,optional) | Use like a support metric ,for identifing the cente coord of Anchor Boxes
        #{this_steps} show how far apart Anchor Boxes centers will be horizontally and vertically
    # - this_offsets (list,optional) | Show how far apart centers of Anchor Boxes will be from uppon left corner
    # - clip_boxes (boolean,optional) | if True , it helps to constrain Anchor Boxes in IMAGE boundaries
    # - variances (list,optional) | The anchor box offset for each coordinate will be divided by its respective variance value
    # - coords (str,optional) | The format of coordinates


class AnchorBoxes(Layer):

    #Input shape : 4D Tensor (batch_size,height,width,channels)

    #Output shape : 5D Tensor (batch, height, width, n_boxes, 8)
    # last axis consists of FOUR anchor box coordinates and the FOUR variance values for each box

    #The main Goal : Create output tensor containing Anchor Box coordinates and vars for based on input tensor


    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5,1.0,2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1,0.1,0.2,0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):

        #НЕ ПРОПИСАНЫ ОШИБКИ!!!!!!

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords

        # Compute the number of boxes per cell. We create 2 boxes for {aspect_ratio}==1
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)

        super(AnchorBoxes, self).__init__(**kwargs)

        # def build(self, input_shape):
        #     self.input_spec = [InputSpec(shape=input_shape)]
        #     super(AnchorBoxes, self).build(input_shape)

        def call(self,x,mask=None):

            #STAGE 1

            # Compute box width and height for each aspect ratio
            # The shorter side of the image will be used to compute width and height using {scale} and {aspect_ratios}

            size=min(self.img_height,self.img_width)

            # Compute the box widths and and heights for all aspect ratios
            wh_list=[]

            for ar in self.aspect_ratios:

                if (ar==1):

                    #Compute the regular anchor box for aspect ratio 1

                    box_height=box_width=self.this_scale*size
                    wh_list.append((box_width,box_height))

                    if self.two_boxes_for_ar1:
                        # Compute one slightly larger versionfor mode {two_boxes_for_ar1}
                        # using the geometric mean of this scale value and the next.
                        box_height=box_width=np.sqrt(self.this_scale*self.naxt_scale)*size
                        wh_list.append((box_width,box_height))

                else:
                    box_height=self.this_scale*size / np.sqrt(ar)
                    box_width=self.this_scale*size * np.sqrt(ar)

            wh_list=np.array(wh_list)

            #STAGE 1 -END

            #STAGE 2

            #Compute the grid of box center points.They ara identical for all aspect ratios
            #Compute the step sizes, i.e how far apart the anchor box center points will
            #be vertically and horizontaly

            if (self.this_steps is None):

                step_height=self.img_height / feature_map_height
                step_width=self.img_width / feature_map_width

            else:
                if isinstance(self.this_steps,(list,tuple)) and (len(self.this_steps) == 2):
                    step_height=self.this_steps[0]
                    step_width=self.thos_steps[1]

                elif isinstance(self.this_steps,(int,float)):

                    step_height=self.this_steps
                    step_width=self.this_steps

            #Compute the offsets , i.e. at what pixel values the first anchor box center point
            #will be from the top and from the left of the image

            if (self.this_offsets is None):

                offset_height = 0.5
                offset_width = 0.5
            else:

                if isinstance(self.this_offsets,(list,tuple)) and (len(self.this_offsets) ==2 ):

                    offset_height = self.this_offsets[0]
                    offset_width = self.this_offsets[1]

                elif isinstance(self.this_offsets,(int,float)):

                    offset_height = self.this_offsets
                    offset_width = self.this_offset

            # STAGE 2 -END

            #STAGE 3

            #Now that we have the offsets and step sizes , compute the grid of anchor
            #box center points

            cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
            cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
            cx_grid, cy_grid = np.meshgrid(cx, cy)

            cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
            cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

            # Create a 4D tensor template of shape (feature_map_height, feature_map_width, n_boxes, 4)
            # where the last dimension will contain (cx, cy, w, h)

            boxes_tensor[:, :, :,0] = np.tile(cx_grid,(1, 1, self.n_boxes)) #Set cx
            boxes_tensor[:, :, :, 1] = np.tile(cy_grid,(1, 1, self.n_boxes)) #Set cy
            boxes_tensor[:, :, :, 2] = wh_listp[:, 0] #Set w
            boxes_tensor[:, :, :, 3] = wh_lsit[:, 1] #Set h

            #Convert (cx,cy,w,h) to (xmin,xmax,ymin,ymax)

            boxes_tensor=conver_coordinates(boxes_tensor,start_index=0,conversion='centroids2corners')

            #if clip_boxes is enabled,clip the coordinates to lie within the image boundaries
            x_coords=boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width -1
            x_coords[x_coords < 0] = 0

            boxes_tensor[:, :, :,[0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height -1
            y_coordsp[y_coords < 0] = 0
            boxes_tensor[:, :, :,[1,3]] = y_coords

            #if normalized_coords is enabled,normalize the coordinates to be within [0,1]
            if self.normalize_coords:
                boxes_tensor[:, :, :,[0, 2]] /= self.img_width
                boxes_tensor[:, :, :,[1, 3]] /= self.img_height

            # Implement box limiting directly for (cx, cy, w, h) so that we don't have to unnecessarily convert back and forth.
            if self.coords == 'centroids':
                # Convert `(xmin, ymin, xmax, ymax)` back to (cx, cy, w, h).
                boxes_tensor = convert_coordinates(boxes_tensor, start_index=0,
                                                   conversion='corners2centroids', border_pixels='half')
            elif self.coords == 'minmax':
                # Convert `(xmin, ymin, xmax, ymax)` to (xmin, xmax, ymin, ymax).
                boxes_tensor = convert_coordinates(boxes_tensor, start_index=0,
                                                   conversion='corners2minmax', border_pixels='half')

            # Create a tensor to contain the variances and append it to boxes_tensor. This tensor has the same shape
            # as boxes_tensor and simply contains the same 4 variance values for every position in the last axis.

            variances_tensor = np.zeros_like(boxes_tensor) # shape (feature_map_height,feature_map_width,n_boxes,8)
            variances_tensor += self.variances

            # Now boxes_tensor becomes a tensor of shape (feature_map_height, feature_map_width, n_boxes, 8)
            boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

            # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`

            boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
            boxes_tensor=tf.Variable(initial_value=boxes_tensor,dtype=tf.float32)
            boxes_tensor=tf.reshape(boxes_tensor,[-1,feature_map_height,feature_map_width,self.n_boxes,8])

            #STAGE 3 -END

            return boxes_tensor






