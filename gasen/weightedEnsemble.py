def weighted_ensemble(weights, models, inputs):
    # weights.shape = 1 x 100
    # inputs.shape = 1 x 200 (example)
    # Assigning empty array to store 2D array of model predictions
    predictions = []

    # Loop through all models
    for i in range(len(models)): # x 100
        # Make predictions for each model
        predictions.append(models[i].predict(inputs)) # 100 x 200
        # Multiply prediction by corresponding weight
        predictions[i] = [x * weights[i] for x in predictions[i]] # 100 x 200 

    # Sum of weighted predictions
    predictionsSum = [sum(x) for x in zip(*predictions)] # x 100 * 200

    return predictionsSum # x 100 * 200
