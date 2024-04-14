import numpy as np

from scipy.spatial.transform import Rotation

# Note: scipy uses quaternions in the form (x, y, z, w) while we use (w, x, y, z).
# Outside of these function definitions, we will observe the (w, x, y, z) ordering.

def convert_rotation_matrix_to_quarternion(matrix):
    rotation = Rotation.from_matrix(matrix)
    q_xyzw = rotation.as_quat()
    q_wxyz = np.roll(q_xyzw, 1)
    return q_wxyz

def convert_quarternion_to_rotation_matrix(q):
    q_xyzw = np.roll(q, -1)
    rotation = Rotation.from_quat(q_xyzw)
    return rotation.as_matrix()
 
"""
Returns the overall effect of applying the transformations in this order:
Rotation R_1, followed by translation T_1, followed by rotation R_2, followed by translation T_2.
"""
def compose_transformations(T_1, R_1, T_2, R_2):
    R_1_xyzw = np.roll(R_1, -1)
    R_2_xyzw = np.roll(R_2, -1)

    R_1 = Rotation.from_quat(R_1_xyzw)
    R_2 = Rotation.from_quat(R_2_xyzw)

    # The overall rotation is simply the product of the two rotations.
    R_overall = R_2 * R_1 # Note the order.
    R_overall_xyzw = R_overall.as_quat()
    R_overall_wxyz = np.roll(R_overall_xyzw, 1)

    # We calculate the changes in position caused by T_1 and T_2, w.r.t. the global frame of reference.
    T_1_global = R_1.apply(T_1) # T_1 is affected by R_1.
    T_2_global = R_overall.apply(T_2) # T_2 is affected by both rotations.
    T_overall_global = T_1_global + T_2_global

    # Then, we take into account the net rotation that has to occur before the translation.
    T_overall = Rotation.inv(R_overall).apply(T_overall_global)

    return T_overall, R_overall_wxyz

"""
Given a transformation (T, R), returns the inverse transformation (T_inverse, R_inverse).
"""
def calculate_inverse_transformation(T, R):
    
    # Determine where the transformation brings a point at the origin to,
    # in terms of the original point of view.
    R_xyzw = np.roll(R, -1)
    R = Rotation.from_quat(R_xyzw)
    T_relative_to_initial_POV = R.apply(T)

    # Now, suppose we wish to undo this transformation.
    # Firstly, we need to undo the rotation.
    R_inverse = R.inv()

    # Afterwards, we have rotated ourselves back to the original point of reference.
    # The translation is simply the negation of the overall translation effect.
    T_inverse = -T_relative_to_initial_POV

    R_inverse_xyzw = R_inverse.as_quat()
    R_inverse_wxyz = np.roll(R_inverse_xyzw, 1)

    return T_inverse, R_inverse_wxyz

"""
Returns the transformation (T, R) which, when applied after
(T_start, R_start), has the same effect as applying the transformation (T_end, R_end).
Note that in any transformation, the rotation is applied first, followed by the translation.
"""
def decompose_transformations(T_start, R_start, T_end, R_end):
    R_start_xyzw = np.roll(R_start, -1)
    R_end_xyzw = np.roll(R_end, -1)

    R_start = R.from_quat(R_start_xyzw)
    R_end = R.from_quat(R_end_xyzw)

    T = T_end - R_end.apply(T_start)
    R = (Rotation.inv(R_start) * R_end).as_quat()
    R = np.roll(R, 1)
    return T, R