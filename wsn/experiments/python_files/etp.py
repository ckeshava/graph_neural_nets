import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
# A and B are two (N by 2) matrices

def func1(p, q, count_invs):
        # print('DEBUG - func1: old count_invs = {}'.format(count_invs))
        # print('I am in func1')
        
        ans = tf.cond(p < q, lambda: add_1_func(), lambda: dummy_func())

        # print('DEBUG - func1: NEW count_invs = {}'.format(ans))

        return ans

def func2(p, q, count_invs):
        # print('DEBUG - func2    : old count_invs = {}'.format(count_invs))
        # print('I am in func2')

        ans = tf.cond(p > q, lambda: add_1_func(), lambda: dummy_func())
        # print('DEBUG - func2: NEW count_invs = {}'.format(ans))

        return ans



def dummy_func():
        # print('I am in dummy_func')
    
        return 0

def add_1_func():
        return 1


def get_etp_without_rotation(A, B):


        n = A.get_shape().as_list()[0]

        A = tf.cast(A, dtype=tf.float32)
        B = tf.cast(B, dtype=tf.float32)


        count_invs = 0


        for i in range(0, n):
        #     print('DEBUG: Presently processing Node:{} / {}'.format(i, n))
                
            for j in range(0, n):


                    # check for inversions along X-axis                        
                    count_invs += tf.cond(A[i, 0] < A[j, 0], lambda: func1(B[j, 0], B[i, 0], count_invs), lambda: dummy_func())
                    # print('DEBUG - MAIN: NEW count_invs = {}'.format(count_invs))

                    count_invs += tf.cond(A[i, 0] > A[j, 0], lambda: func2(B[j, 0], B[i, 0], count_invs), lambda: dummy_func())
                    # print('DEBUG - MAIN: NEW count_invs = {}'.format(count_invs))

                    # check for inversions along Y-axis
                    count_invs += tf.cond(A[i, 1] < A[j, 1], lambda: func1(B[j, 1], B[i, 1], count_invs), lambda: dummy_func())
                    # print('DEBUG - MAIN: NEW count_invs = {}'.format(count_invs))

                    count_invs += tf.cond(A[i, 1] > A[j, 1], lambda: func2(B[j, 1], B[i, 1], count_invs), lambda: dummy_func())
                    # print('DEBUG - MAIN: NEW count_invs = {}'.format(count_invs))



        count_invs/=(n * (n - 1))
        count_invs *= 100
        return count_invs

# A is true value
# B is predicted value
def get_best_etp(A, B):
        A = tf.reshape(A, shape=(-1, 2))
        B = tf.reshape(B, shape=(-1, 2))


        print('Number of nodes in the Graph: {}'.format(A.shape[0]))

        x = 0
        final_etp = get_etp_without_rotation(A, B)
#         for x in range(0, 360, 100):
# #                 print('Rotation: {}'.format(x))
#                 rot_matrix = np.array([[np.cos(np.radians(x)), -np.sin(np.radians(x))], [np.sin(np.radians(x)), np.cos(np.radians(x))]])
#                 rot_matrix = tf.convert_to_tensor(rot_matrix, dtype=tf.float32)
#                 # temp = get_etp_without_rotation(A, tf.matmul(B, tf.cast(rot_matrix, dtype=tf.float64)))
#                 A = tf.cast(A, dtype=tf.float32)
#                 B = tf.cast(B, dtype=tf.float32)
#                 temp = get_etp_without_rotation(A, tf.matmul(B, rot_matrix))

#                 final_etp = tf.cond(temp > final_etp, lambda: temp, lambda: tf.cast(final_etp, dtype=tf.float64))

#                 # if x % 10 == 0:
#                 # print('Current best of count_invs = {} at Angle {}'.format(final_etp, x))

                
        return final_etp
