def generator(self, inputs, mode='train', reuse=False, scope='g_net'):
        var_trainable=False
        if (mode=='train'):
            var_trainable=True

        with tf.compat.v1.variable_scope(scope):
            with tf.compat.v1.variable_scope('pre_conv'):
                conv_pre = conv2d_bn(inputs[0], 64, 3, activation=None, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, is_training=self.is_training, trainable=var_trainable, scope='conv_pre')
            
            with tf.compat.v1.variable_scope('guide_upsample'):
                conv1_up_guide = conv2d_bn(inputs[1], 1, 3, activation=tf.nn.relu, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, is_training=self.is_training, trainable=var_trainable, scope='conv1_up_guide')
                conv2_up_guide = conv2d_bn(conv1_up_guide, 1, 3, activation=tf.nn.relu, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, is_training=self.is_training, trainable=var_trainable, scope='conv2_up_guide')
            
            with tf.compat.v1.variable_scope('pre_conv_guide'):
                conv_pre_guide = conv2d_bn(conv2_up_guide, 64, 3, activation=None, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, is_training=self.is_training, trainable=var_trainable, scope='conv_pre_guide')

            with tf.compat.v1.variable_scope('encoder'):
                conv1 = ResBlock_bn(conv_pre, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock1')
                conv2 = ResBlock_bn(conv1, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock2')
                conv3 = ResBlock_bn(conv2, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock3')
                
            with tf.compat.v1.variable_scope('encoder_guide'):
                conv1_guide = ResBlock_bn(conv_pre_guide, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock1')
                conv2_guide = ResBlock_bn(conv1_guide, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock2')
                conv3_guide = ResBlock_bn(conv2_guide, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock3')
           
            with tf.compat.v1.variable_scope('dec1'):
                conv4_inp = feature_fusing(conv3, conv3_guide, 64, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='fuse')
                conv4 = ResBlock_bn(conv4_inp, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock4')
                inc4 = conv4 + conv2
                
            with tf.compat.v1.variable_scope('dec2'):
                conv5_inp = feature_fusing(inc4, conv2_guide, 64, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='fuse')
                conv5 = ResBlock_bn(conv5_inp, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock5')
                inc5 = conv5 + conv1

            with tf.compat.v1.variable_scope('dec3'):
                conv6_inp = feature_fusing(inc5, conv1_guide, 64, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='fuse')
                conv6 = ResBlock_bn(conv6_inp, 64, 3, activate_residual=True, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, is_training=self.is_training, scope='Resblock6')
                inc6 = conv6 + conv_pre

            with tf.compat.v1.variable_scope('post_conv'):
                inc6 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(inc6, training=self.is_training, trainable=var_trainable, reuse=reuse))
                conv7 = conv2d(inc6, 16, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, scope='conv1')
                out_pred = conv2d(conv7, self.chns, 3, activation=None, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=reuse, trainable=var_trainable, scope='conv2')
                out_img = out_pred + inputs[0]

        return tf.concat([out_img, out_img], axis=3)