#     @staticmethod
#     def _updateTvec(tvec, obj_center=None, distance=None):
#         '''
#         obj_center (x,y) - center of plane [px]
#         distance [px]
#         '''
#         if distance is not None:
#             tvec[2,0]=distance
#         if obj_center is not None:
#             tvec[0,0]=obj_center[0]
#             tvec[1,0]=obj_center[1]
#         return tvec
    
#     @staticmethod
#     def _updateRVec(rvec, rotation=None, tilt=None):
#         '''
#         in DEG
#         '''
#         if rotation is None and tilt is None:
#             return rvec
#         r, t, p = rvec2euler(rvec)
#         if rotation is not None:
#             r = np.radians(rotation)
#         if tilt is not None:
#             t =  np.radians(tilt)  
#         return euler2rvec(r,t,p)

# 
#     def corrDepthMap(self, midpointdepth=0, pose=None):
# 
#         if pose is None:
#             pose = self.pose()
#         tvec,rvec = pose
# 
# 
# #         tvec = self._updateTvec(tvec, obj_center, distance)
# #         rvec = self._updateRVec(rvec, rotation, tilt)
# 
#         #with current/given obj. rotation:
#         d = self.depthMap( pose=(tvec,rvec))
# #         return d+midpointdepth
# #         return d
#         #without object rotation:
#         
#         ##################
#         #middle point has to be img middle when zero rot
#         ####################
#         
# #         diag = np.linalg.norm(self.quad[0,0]-self.objCenter())
# #         f = diag / np.linalg.norm(self.obj_points[0,:2])
#         
#         #no-tilt rotation vector:
#         rvec0 = self._updateRVec(rvec, tilt=0)
# #         r,_,p = rvec2euler(rvec)
# #         rvec0 = euler2rvec(r,0,p)
# 
#         d2 = self.depthMap( (tvec,rvec0))
#         dd = d-d2
# #         dd = d
# 
# #         if self.px_per_phys_unit is None:
# #             objwidth = self.obj_points[1,0]*2
# #             pxwidth = np.linalg.norm(self.quad[1]-self.quad[0])
# #             self.px_per_phys_unit = pxwidth / objwidth
#         
#         #add depth at object mid point:
#         x,y = self.objCenter()
#         v = dd[int(y),int(x) ]
#         dd+=midpointdepth-v
#         
# #         if excludeBG:
# #             bg = np.ones_like(dd, dtype=np.uint8)
# #             cv2.fillConvexPoly(bg, self.quad, 0)
# #             bg = bg.astype(bool)
# #             import pylab as plt
# #             plt.imshow(bg)
# #             plt.show()
#             
#         return dd











### remove as soon as test works       
#         def vectorAngle(vec1,vec2):
#             a = np.arccos(
#                         np.einsum('ijk,ijk->jk', vec1,vec2)
#                         /( norm(vec1,axis=0) * norm(vec2,axis=0) ) )
#             #take smaller of both possible angles:
#             ab = np.abs(np.pi-a)
#             with np.errstate(invalid='ignore'):
#                 i = a>ab
#             a[i] = ab[i]
#             return a
#         
#         def vectorToField(vec):
#             out = np.empty(shape=(3,s0,s1))
#             out[0] = vec[0]
#             out[1] = vec[1]
#             out[2] = vec[2]
#             return out
###



#         dd = kwargs['midpointdepth']
        #unify deph from focal point:
#         vv0 = np.empty_like(v0)
#         vv0[:2]=v0[:2]
#         vv0[2]=dd

#         beta1 = vectorAngle(vv0, vectorToField([0.,0.,1.]) )


















#         beta2 -= np.pi/2
#         beta2 = np.abs(beta2)

        #length of one pixel in focal plane
#         size_px_x, size_px_y = v0[:2,1, 1]-v0[:2,0,0]
#         print (v0[:2,1, 1], v0[:2,0,0], size_px_x, size_px_y )
#         size_px_x, size_px_y = 3.7,3.7
     
        #area on plane of one image pixel
        #using theorem on intersecting lines:
#         A = size_px_x * size_px_y * (s/dd)**2


#         dy = v0[1,0,0]-v0[1,-1,0]
#         dz = v0[2,0,0]-v0[2,-1,0]
#         dx = v0[0,0,0]-v0[0,-1,0]
      
# 
#         beta2[~self.foreground()]=np.nan

#         import pylab as plt
#         from mpl_toolkits.mplot3d import Axes3D
# # # #         
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')        
#         ax.plot_surface(v0[0,::10,::10], v0[1,::10,::10], v0[2,::10,::10])
#         plt.show()
# #         
#         plt.imshow(np.degrees(beta2))
#         plt.colorbar()
#         plt.figure(2)
#         plt.imshow(np.degrees(beta1))
#         plt.colorbar()
# # #   
#         plt.figure(3)
#         plt.imshow(v0[0])
#         plt.colorbar()
# # # # #   
#         plt.figure(4)
#         plt.imshow(v0[1])
#         plt.colorbar()
# # # #    
#         plt.show()
 
#         vf = (dA*np.cos(beta1)*np.cos(beta2)) / s**2
        #beta2[:]=beta2.mean()
#         vf = (A*np.cos(beta2)) / s**2

#         import pylab as plt
#         plt.figure(4)
#         plt.imshow(vf)
#         plt.colorbar()
# # #    
#         plt.show()







    #TODO: remove!
#     def _getTiltFactor(self, shape, orientation=None):
#         '''
#         optional:
#         orientation = ( distance to image plane in px,
#                         object tilt [radians],
#                         object rotation [radians] )
#         otherwise calculated from homography
#         '''
#         #CALCULATE VIGNETTING OF WARPED OBJECT:
# #         f = self.opts['cameraMatrix'][0,0]
#         try: dist, tilt, rot = orientation
#         except TypeError:
#             dist, tilt, rot = self.objectOrientation()
#             rot -= np.pi/2# because of different references
#         if self.quad is not None:
#             cent = self.quad[:,1].mean(), self.quad[:,0].mean()
#         else:
#             cent = None
# #         self.maps['tilt_factor'] = 
#         tf = np.fromfunction(lambda x,y: tiltFactor((x, y), 
#                                               f=dist, tilt=tilt, rot=rot, center=cent), shape[:2])
#         #if img is color:
#         if tf.shape != shape:
#             tf = np.repeat(tf, shape[-1]).reshape(shape)
#         return tf