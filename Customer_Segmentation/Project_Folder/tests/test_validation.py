import math

import pytest


# ---------------------------------------
# Limits (match the HTML form)
# ---------------------------------------

MIN_AREA = 100
MAX_AREA = 10000

MIN_ROOMS = 1
MAX_ROOMS = 10

MIN_AGE = 0
MAX_AGE = 100


# ---------------------------------------
# Helpers
# ---------------------------------------

def _is_number(value):

	return (
		isinstance(value, (int, float))
		and not isinstance(value, bool)
		and math.isfinite(value)
	)


def _check_rooms(label, value, errors):

	if not _is_number(value):
		errors.append(f"{label} must be a valid number.")
		return

	if value != int(value):
		errors.append(f"{label} must be a whole number.")
		return

	if not MIN_ROOMS <= value <= MAX_ROOMS:
		errors.append(
			f"{label} must be between {MIN_ROOMS} and {MAX_ROOMS}."
		)


# ---------------------------------------
# Validate House Data
# ---------------------------------------

def validate_house_data(
	area,
	bedrooms,
	bathrooms,
	age
):

	errors = []

	# Area

	if not _is_number(area):
		errors.append("Area must be a valid number.")

	elif not MIN_AREA <= area <= MAX_AREA:
		errors.append(
			f"Area must be between {MIN_AREA} and {MAX_AREA} sqft."
		)

	# Bedrooms and bathrooms

	_check_rooms("Bedrooms", bedrooms, errors)

	_check_rooms("Bathrooms", bathrooms, errors)

	# Age

	if not _is_number(age):
		errors.append("House age must be a valid number.")

	elif not MIN_AGE <= age <= MAX_AGE:
		errors.append(
			f"House age must be between {MIN_AGE} and {MAX_AGE} years."
		)

	return errors


# =======================================
# TESTS
# =======================================


# ---------------------------------------
# Valid input
# ---------------------------------------

def test_valid_input_returns_no_errors():

	assert validate_house_data(1350, 3, 2, 6) == []


def test_float_values_for_area_and_age_are_allowed():

	assert validate_house_data(1350.5, 3, 2, 6.5) == []


def test_whole_number_floats_for_rooms_are_allowed():

	assert validate_house_data(1350, 3.0, 2.0, 6) == []


@pytest.mark.parametrize("area, bedrooms, bathrooms, age", [
	(100, 1, 1, 0),          # lower limits
	(10000, 10, 10, 100),    # upper limits
])
def test_boundary_values_are_valid(area, bedrooms, bathrooms, age):

	assert validate_house_data(area, bedrooms, bathrooms, age) == []


# ---------------------------------------
# Area
# ---------------------------------------

@pytest.mark.parametrize("area", [0, -500, 99, 10001])
def test_area_out_of_range(area):

	errors = validate_house_data(area, 3, 2, 6)

	assert len(errors) == 1
	assert "Area" in errors[0]


# ---------------------------------------
# Bedrooms and bathrooms
# ---------------------------------------

@pytest.mark.parametrize("bedrooms", [0, -1, 11])
def test_bedrooms_out_of_range(bedrooms):

	errors = validate_house_data(1350, bedrooms, 2, 6)

	assert len(errors) == 1
	assert "Bedrooms" in errors[0]


@pytest.mark.parametrize("bathrooms", [0, -1, 11])
def test_bathrooms_out_of_range(bathrooms):

	errors = validate_house_data(1350, 3, bathrooms, 6)

	assert len(errors) == 1
	assert "Bathrooms" in errors[0]


def test_rooms_must_be_whole_numbers():

	errors = validate_house_data(1350, 2.5, 1.5, 6)

	assert len(errors) == 2
	assert all("whole number" in e for e in errors)


# ---------------------------------------
# Age
# ---------------------------------------

@pytest.mark.parametrize("age", [-1, 101, 250])
def test_age_out_of_range(age):

	errors = validate_house_data(1350, 3, 2, age)

	assert len(errors) == 1
	assert "age" in errors[0].lower()


# ---------------------------------------
# Invalid types and special values
# ---------------------------------------

@pytest.mark.parametrize("bad", [None, "abc", "", float("nan"), float("inf"), True])
def test_non_numeric_values_are_rejected(bad):

	assert len(validate_house_data(bad, 3, 2, 6)) == 1
	assert len(validate_house_data(1350, bad, 2, 6)) == 1
	assert len(validate_house_data(1350, 3, bad, 6)) == 1
	assert len(validate_house_data(1350, 3, 2, bad)) == 1


# ---------------------------------------
# Multiple errors
# ---------------------------------------

def test_multiple_errors_are_all_reported():

	errors = validate_house_data(-10, 0, 20, 500)

	assert len(errors) == 4


def test_returns_a_list():

	assert isinstance(validate_house_data(1350, 3, 2, 6), list)