import numpy as np

from scipy.spatial.transform import Rotation

# Note: scipy uses quaternions in the form (x, y, z, w) while we use (w, x, y, z).
# Outside of these function definitions, we will observe the (w, x, y, z) ordering.
 
"""
Returns the overall effect of applying the transformations in this order:
Rotation R_1, followed by translation T_1, followed by rotation R_2, followed by translation T_2.
"""
def compose_transformations(R_1, T_1, R_2, T_2):
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

    return R_overall_wxyz, T_overall

"""
Given a transformation (R, T), returns the inverse transformation (R_inverse, T_inverse).
"""
def calculate_inverse_transformation(R, T):
    
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

    return R_inverse_wxyz, T_inverse

"""
Takes in two transformations A_to_B and A_to_C.
Returns the transformation B_to_C, which, as its name implies, is such that
applying A_to_B, then B_to_C to a spacecraft has the same overall effect as applying A_to_C as the spacecraft.

Note again that in any transformation, the rotation is applied first, followed by the translation.
"""
def decompose_transformations(A_to_B_rotation, A_to_B_translation, A_to_C_rotation, A_to_C_translation):
    # We first calculate the inverse of the starting transformation.
    B_to_A_rotation, B_to_A_translation = calculate_inverse_transformation(A_to_B_rotation, A_to_B_translation)

    # Then, we compose the inverse of the starting transformation with the ending transformation.
    B_to_C_rotation, B_to_C_translation = compose_transformations(B_to_A_rotation, B_to_A_translation,
                                                                  A_to_C_rotation, A_to_C_translation)

    return B_to_C_rotation, B_to_C_translation