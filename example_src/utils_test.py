from utils import *
import numpy as np
from scipy.spatial.transform import Rotation

# The tests can be run by running `pytest` in the terminal in the directory containing this file.

IDENTITY_TRANSLATION = np.zeros((3,))
IDENTITY_ROTATION = np.array([1, 0, 0, 0])

def test_identity_composed_with_identity_is_identity():
    R, T = compose_transformations(IDENTITY_ROTATION, IDENTITY_TRANSLATION, IDENTITY_ROTATION, IDENTITY_TRANSLATION)
    assert np.allclose(T, IDENTITY_TRANSLATION)
    assert np.allclose(R, IDENTITY_ROTATION)

def test_inverse_of_identity_is_identity():
    R, T = calculate_inverse_transformation(IDENTITY_ROTATION, IDENTITY_TRANSLATION)
    assert np.allclose(T, IDENTITY_TRANSLATION)
    assert np.allclose(R, IDENTITY_ROTATION)

def test_inverse_of_random_translation():
    T = np.random.rand(3)
    R = IDENTITY_ROTATION
    R_inv, T_inv = calculate_inverse_transformation(R, T)
    assert np.allclose(T_inv, -T)
    assert np.allclose(R_inv, IDENTITY_ROTATION)

def test_inverse_of_random_rotation():
    T = IDENTITY_TRANSLATION
    R_wxyz = np.random.rand(4)
    R_wxyz /= np.linalg.norm(R_wxyz) # Ensure that the quaternion R is a unit quaternion.
    actual_R_inv_wxyz, actual_T_inv = calculate_inverse_transformation(R_wxyz, T)

    # Calculate the expected results.
    R_xyzw = np.roll(R_wxyz, -1)
    R = Rotation.from_quat(R_xyzw)
    expected_T_inv = IDENTITY_TRANSLATION
    expected_R_inv_xyzw = R.inv().as_quat()
    expected_R_inv_wxyz = np.roll(expected_R_inv_xyzw, 1)
    
    assert np.allclose(actual_T_inv, expected_T_inv)
    assert np.allclose(actual_R_inv_wxyz, expected_R_inv_wxyz)

def test_inverse_of_known_transformation():
    # Consider a spaceship (with the camera) at the origin, and whose orientation is the identity.
    # In this test case, we move the spaceship in the x-y plane (z = 0).

    # We rotate the spaceship by 90 degrees about the z axis.
    R_xyzw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # If we wish to undo this transformation,...
    
    # We first rotate the spaceship by -90 degrees about the z axis.
    R_inv_expected_xyzw = Rotation.from_euler('z', -90, degrees=True).as_quat()
    R_inv_expected_wxyz = np.roll(R_inv_expected_xyzw, -1)

    # Now, the spaceship is at (0, -5, 0).
    # Hence, we should move the spaceship 5 units in its y direction.
    # This is also the global y direction.
    T_inv_expected = np.array([0, 5, 0])

    R_inv_actual_wxyz, T_inv_actual = calculate_inverse_transformation(R_wxyz, T)

    assert np.allclose(T_inv_actual, T_inv_expected, atol=1e-5)
    assert np.allclose(R_inv_actual_wxyz, R_inv_expected_wxyz, atol=1e-5)

def test_compose_known_transformations():
    # Consider a spaceship at the origin and whose orientation is the identity.
    # We rotate the spaceship by 90 degrees about the z axis.
    R_xyzw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # We then, again, rotate the spaceship by 90 degrees about the z axis.
    # We then, again, move the spaceship 5 units in its x direction.
    # This corresponds to the global negative x direction.

    # The overall effect is that the spaceship is at (-5, -5, 0) and is oriented at 180 degrees about the z axis.
    R_overall_expected_xyzw = Rotation.from_euler('z', 180, degrees=True).as_quat()
    R_overall_expected_wxyz = np.roll(R_overall_expected_xyzw, -1)
    T_overall_expected = np.array([5, 5, 0]) # Because the spaceship is 'upside down' in the x-y plane.

    R_overall_actual_wxyz, T_overall_actual = compose_transformations(R_wxyz, T, R_wxyz, T)

    assert np.allclose(R_overall_actual_wxyz, R_overall_expected_wxyz, atol=1e-5)
    assert np.allclose(T_overall_actual, T_overall_expected, atol=1e-5)

def test_compose_random_translation_with_random_translation():
    T1 = np.random.rand(3)
    T2 = np.random.rand(3)
    R_overall, T_overall = compose_transformations(IDENTITY_ROTATION, T1, IDENTITY_ROTATION, T2)
    assert np.allclose(T_overall, T1 + T2)
    assert np.allclose(R_overall, IDENTITY_ROTATION)

def test_compose_random_rotation_with_random_rotation():
    R1_wxyz = np.random.rand(4)
    R2_wxyz = np.random.rand(4)
    R1_wxyz /= np.linalg.norm(R1_wxyz) # Ensure that the quaternion is a unit quaternion.
    R2_wxyz /= np.linalg.norm(R2_wxyz) # Ensure that the quaternion is a unit quaternion.
    R1_xyzw = np.roll(R1_wxyz, -1)
    R2_xyzw = np.roll(R2_wxyz, -1)
    R1 = Rotation.from_quat(R1_xyzw)
    R2 = Rotation.from_quat(R2_xyzw)
    R_overall, T_overall = compose_transformations(R1_wxyz, IDENTITY_TRANSLATION, R2_wxyz, IDENTITY_TRANSLATION)
    R_overall_expected = R2 * R1
    R_overall_expected_xyzw = R_overall_expected.as_quat()
    R_overall_expected_wxyz = np.roll(R_overall_expected_xyzw, 1)
    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert np.allclose(R_overall, R_overall_expected_wxyz)

def test_compose_transformation_with_known_inverse():
    # Consider a spaceship (with the camera) at the origin, and whose orientation is the identity.
    # In this test case, we move the spaceship in the x-y plane (z = 0).

    # We rotate the spaceship by 90 degrees about the z axis.
    R_xyzw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # If we wish to undo this transformation,...
    
    # We first rotate the spaceship by -90 degrees about the z axis.
    R_inv_xyzw = Rotation.from_euler('z', -90, degrees=True).as_quat()
    R_inv_wxyz = np.roll(R_inv_xyzw, -1)

    # Now, the spaceship is at (0, -5, 0).
    # Hence, we should move the spaceship 5 units in its y direction.
    # This is also the global y direction.
    T_inv = np.array([0, 5, 0])

    R_overall_wxyz, T_overall = compose_transformations(R_wxyz, T, R_inv_wxyz, T_inv)

    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert np.allclose(R_overall_wxyz, IDENTITY_ROTATION)


def test_transformation_composed_with_inverse_is_identity():
    np.random.seed(0)
    T = np.random.rand(3)
    R = np.random.rand(4)
    R /= np.linalg.norm(R) # Ensure that the quaternion R is a unit quaternion.
    R_inv, T_inv = calculate_inverse_transformation(R, T)
    R_overall, T_overall = compose_transformations(R, T, R_inv, T_inv)
    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert np.allclose(R_overall, IDENTITY_ROTATION)