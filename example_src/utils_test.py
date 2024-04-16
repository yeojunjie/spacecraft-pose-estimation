from utils import *
import numpy as np
from scipy.spatial.transform import Rotation

# The tests can be run by running `pytest` in the terminal in the directory containing this file.

IDENTITY_TRANSLATION = np.zeros((3,))
IDENTITY_ROTATION = np.array([1, 0, 0, 0])
np.random.seed(1)

def generate_random_rotation_quaternion():
    # Note: Whether this is in wxyz or xyzw format does not matter.
    R = np.random.rand(4) * 2 - 1 # rand returns values in [0, 1). So, we scale to [-1, 1).
    R /= np.linalg.norm(R) # Ensure that the quaternion is a unit quaternion.
    return R

def generate_random_translation():
    return np.random.rand(3) * 10 - 5 # rand returns values in [0, 1). So, we scale to [-5, 5).

def quaternions_are_equal(q1, q2):
    return np.allclose(q1, q2) or np.allclose(q1, -q2)

def test_identity_composed_with_identity_is_identity():
    R, T = compose_transformations(IDENTITY_ROTATION, IDENTITY_TRANSLATION, IDENTITY_ROTATION, IDENTITY_TRANSLATION)
    assert np.allclose(T, IDENTITY_TRANSLATION)
    assert quaternions_are_equal(R, IDENTITY_ROTATION)

def test_inverse_of_identity_is_identity():
    R, T = calculate_inverse_transformation(IDENTITY_ROTATION, IDENTITY_TRANSLATION)
    assert np.allclose(T, IDENTITY_TRANSLATION)
    assert quaternions_are_equal(R, IDENTITY_ROTATION)

def test_inverse_of_random_translation():
    T = generate_random_translation()
    R = IDENTITY_ROTATION
    R_inv, T_inv = calculate_inverse_transformation(R, T)
    assert np.allclose(T_inv, -T)
    assert quaternions_are_equal(R_inv, R)

def test_inverse_of_random_rotation():
    T = IDENTITY_TRANSLATION
    R_wxyz = generate_random_rotation_quaternion()
    actual_R_inv_wxyz, actual_T_inv = calculate_inverse_transformation(R_wxyz, T)

    # Calculate the expected results.
    R_xyzw = np.roll(R_wxyz, -1)
    R = Rotation.from_quat(R_xyzw)
    expected_T_inv = IDENTITY_TRANSLATION
    expected_R_inv_xyzw = R.inv().as_quat()
    expected_R_inv_wxyz = np.roll(expected_R_inv_xyzw, 1)
    
    assert np.allclose(actual_T_inv, expected_T_inv)
    assert quaternions_are_equal(actual_R_inv_wxyz, expected_R_inv_wxyz)

def test_inverse_of_known_transformation():
    # Consider a spaceship (with the camera) at the origin, and whose orientation is the identity.
    # In this test case, we move the spaceship in the x-y plane (z = 0).

    # We rotate the spaceship by 90 degrees clockwise about the z axis.
    R_xyzw = Rotation.from_euler('z', -90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # If we wish to undo this transformation,...
    
    # We first rotate the spaceship by 90 degrees anticlockwise about the z axis.
    R_inv_expected_xyzw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    R_inv_expected_wxyz = np.roll(R_inv_expected_xyzw, 1)

    # Now, the spaceship is at (0, -5, 0).
    # Hence, we should move the spaceship 5 units in its y direction.
    # This is also the global y direction.
    T_inv_expected = np.array([0, 5, 0])

    R_inv_actual_wxyz, T_inv_actual = calculate_inverse_transformation(R_wxyz, T)

    assert np.allclose(T_inv_actual, T_inv_expected, atol=1e-5)
    assert quaternions_are_equal(R_inv_actual_wxyz, R_inv_expected_wxyz)

def test_compose_known_transformations():
    # Consider a spaceship at the origin and whose orientation is the identity.
    # We rotate the spaceship by 90 degrees clockwise about the z axis.
    R_xyzw = Rotation.from_euler('z', -90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # We then, again, rotate the spaceship by 90 degrees clockwise about the z axis.
    # We then, again, move the spaceship 5 units in its x direction.
    # This corresponds to the global negative x direction.

    # The overall effect is that the spaceship is at (-5, -5, 0) and is oriented at 180 degrees about the z axis.
    R_overall_expected_xyzw = Rotation.from_euler('z', 180, degrees=True).as_quat()
    R_overall_expected_wxyz = np.roll(R_overall_expected_xyzw, 1)
    T_overall_expected = np.array([5, 5, 0]) # Because the spaceship is 'upside down' in the x-y plane.

    R_overall_actual_wxyz, T_overall_actual = compose_transformations(R_wxyz, T, R_wxyz, T)

    # TODO: The following assertion fails because it does not understand that
    # q === -q. This is because the quaternion representation is not unique.
    assert quaternions_are_equal(R_overall_actual_wxyz, R_overall_expected_wxyz)
    assert np.allclose(T_overall_actual, T_overall_expected, atol=1e-5)

def test_compose_random_translation_with_random_translation():
    T1 = generate_random_translation()
    T2 = generate_random_translation()
    R_overall, T_overall = compose_transformations(IDENTITY_ROTATION, T1, IDENTITY_ROTATION, T2)
    assert np.allclose(T_overall, T1 + T2)
    assert quaternions_are_equal(R_overall, IDENTITY_ROTATION)

def test_compose_random_rotation_with_random_rotation():
    R1_wxyz = generate_random_rotation_quaternion()
    R2_wxyz = generate_random_rotation_quaternion()

    # Use scipy to convert the quaternions to Rotation objects.
    R1_xyzw = np.roll(R1_wxyz, -1)
    R2_xyzw = np.roll(R2_wxyz, -1)
    R1 = Rotation.from_quat(R1_xyzw)
    R2 = Rotation.from_quat(R2_xyzw)
    
    R_overall, T_overall = compose_transformations(R1_wxyz, IDENTITY_TRANSLATION, R2_wxyz, IDENTITY_TRANSLATION)
    R_overall_expected = R2 * R1
    R_overall_expected_xyzw = R_overall_expected.as_quat()
    R_overall_expected_wxyz = np.roll(R_overall_expected_xyzw, 1)
    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert quaternions_are_equal(R_overall, R_overall_expected_wxyz)

def test_compose_transformation_with_known_inverse():
    # Consider a spaceship (with the camera) at the origin, and whose orientation is the identity.
    # In this test case, we move the spaceship in the x-y plane (z = 0).

    # We rotate the spaceship by 90 degrees clockwise about the z axis.
    R_xyzw = Rotation.from_euler('z', -90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # If we wish to undo this transformation,...
    
    # We first rotate the spaceship by 90 degrees anticlockwise about the z axis.
    R_inv_xyzw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    R_inv_wxyz = np.roll(R_inv_xyzw, 1)

    # Now, the spaceship is at (0, -5, 0).
    # Hence, we should move the spaceship 5 units in its y direction.
    # This is also the global y direction.
    T_inv = np.array([0, 5, 0])

    R_overall_wxyz, T_overall = compose_transformations(R_wxyz, T, R_inv_wxyz, T_inv)

    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert quaternions_are_equal(R_overall_wxyz, IDENTITY_ROTATION)


def test_transformation_composed_with_inverse_is_identity():
    T = generate_random_translation()
    R = generate_random_rotation_quaternion()
    R_inv, T_inv = calculate_inverse_transformation(R, T)
    R_overall, T_overall = compose_transformations(R, T, R_inv, T_inv)
    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert quaternions_are_equal(R_overall, IDENTITY_ROTATION)

def test_decompose_transformations_using_inverse_transformations():
    # Generate a random transformation A_to_B.
    A_to_B_rotation = generate_random_rotation_quaternion()
    A_to_B_translation = generate_random_translation()

    # We are testing the decomposition function, so we can assume that the transformation inversion function is correct.
    expected_B_to_A_rotation, expected_B_to_A_translation = calculate_inverse_transformation(A_to_B_rotation, A_to_B_translation)

    # If we apply A_to_B, then B_to_A, it should be the same as applying the identity transformation.
    actual_B_to_A_rotation, actual_B_to_A_translation = decompose_transformations(A_to_B_rotation, A_to_B_translation,
                                                                                  IDENTITY_ROTATION, IDENTITY_TRANSLATION)
    
    assert quaternions_are_equal(actual_B_to_A_rotation, expected_B_to_A_rotation)
    assert np.allclose(actual_B_to_A_translation, expected_B_to_A_translation)

def test_decompose_transformations_using_random_transformations():

    # Generate two random transformations, A_to_B and A_to_C.
    A_to_B_rotation = generate_random_rotation_quaternion()
    A_to_B_translation = generate_random_translation()
    A_to_C_rotation = generate_random_rotation_quaternion()
    A_to_C_translation = generate_random_translation()

    # Calculate the B_to_C transformation.
    B_to_C_rotation, B_to_C_translation = decompose_transformations(A_to_B_rotation, A_to_B_translation,
                                                                    A_to_C_rotation, A_to_C_translation)
    
    # If we apply A_to_B, then B_to_C, it should be the same as applying A_to_C.
    # We are testing the decomposition function, so we can assume that the transformation composition function is correct.
    should_equal_A_to_C_rotation, should_equal_A_to_C_translation = compose_transformations(A_to_B_rotation, A_to_B_translation,
                                                                                            B_to_C_rotation, B_to_C_translation)
    
    assert quaternions_are_equal(should_equal_A_to_C_rotation, A_to_C_rotation)
    assert np.allclose(should_equal_A_to_C_translation, A_to_C_translation)