def validate_house_data(
	area,
	bedrooms,
	bathrooms,
	age
):

	errors = []


	# Area

	if area <= 0:
		errors.append(
			"Area must be greater than 0."
		)

	if area > 10000:
		errors.append(
			"Area cannot exceed 10,000 sqft."
		)


	# Bedrooms

	if bedrooms < 1:
		errors.append(
			"Bedrooms must be at least 1."
		)

	if bedrooms > 10:
		errors.append(
			"Bedrooms cannot exceed 10."
		)


	# Bathrooms

	if bathrooms < 1:
		errors.append(
			"Bathrooms must be at least 1."
		)

	if bathrooms > 10:
		errors.append(
			"Bathrooms cannot exceed 10."
		)


	# Age

	if age < 0:
		errors.append(
			"Age cannot be negative."
		)

	if age > 100:
		errors.append(
			"Age cannot exceed 100 years."
		)


	return errors
