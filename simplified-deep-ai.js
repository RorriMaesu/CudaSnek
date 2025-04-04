// Simplified deep learning AI implementation (no TensorFlow dependency)
class SimplifiedDeepAI {
    constructor(gridWidth, gridHeight) {
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;

        // Initialize neural network weights
        this.initializeWeights();

        // Learning parameters
        this.learningRate = 0.01;
        this.explorationRate = 0.1;

        // Experience replay memory for training
        this.replayMemory = [];
        this.maxMemorySize = 10000;
        this.batchSize = 64;

        // Training statistics
        this.trainingCount = 0;
        this.totalLoss = 0;
        this.averageLoss = 0;

        // Last state and action for training
        this.lastState = null;
        this.lastAction = null;
        this.lastHiddenLayer = null;
    }

    // Initialize neural network weights
    initializeWeights() {
        // Input to hidden layer weights (11 inputs x 8 hidden neurons)
        this.inputToHidden = Array(11).fill().map(() =>
            Array(8).fill().map(() => Math.random() * 2 - 1)
        );

        // Hidden to output layer weights (8 hidden neurons x 4 outputs)
        this.hiddenToOutput = Array(8).fill().map(() =>
            Array(4).fill().map(() => Math.random() * 2 - 1)
        );

        // Bias terms
        this.hiddenBias = Array(8).fill().map(() => Math.random() * 2 - 1);
        this.outputBias = Array(4).fill().map(() => Math.random() * 2 - 1);

        console.log('Initialized deep learning weights');
    }

    // Forward pass through the neural network
    forward(state, currentDirection) {
        // Convert state to array format
        const inputArray = [
            state.foodDirX,
            state.foodDirY,
            state.dangerUp,
            state.dangerRight,
            state.dangerDown,
            state.dangerLeft,
            state.loopDetected ? 1 : 0,
            currentDirection === 'up' ? 1 : 0,
            currentDirection === 'right' ? 1 : 0,
            currentDirection === 'down' ? 1 : 0,
            currentDirection === 'left' ? 1 : 0
        ];

        // Hidden layer
        const hiddenLayer = Array(8).fill(0);
        for (let i = 0; i < 8; i++) {
            // Sum weighted inputs
            for (let j = 0; j < inputArray.length; j++) {
                hiddenLayer[i] += inputArray[j] * this.inputToHidden[j][i];
            }
            // Add bias
            hiddenLayer[i] += this.hiddenBias[i];
            // Apply ReLU activation function
            hiddenLayer[i] = Math.max(0, hiddenLayer[i]);
        }

        // Output layer
        const outputLayer = Array(4).fill(0);
        for (let i = 0; i < 4; i++) {
            // Sum weighted inputs from hidden layer
            for (let j = 0; j < hiddenLayer.length; j++) {
                outputLayer[i] += hiddenLayer[j] * this.hiddenToOutput[j][i];
            }
            // Add bias
            outputLayer[i] += this.outputBias[i];
        }

        // Apply softmax to get probabilities
        const expValues = outputLayer.map(x => Math.exp(x));
        const expSum = expValues.reduce((a, b) => a + b, 0);
        const actionProbabilities = expValues.map(x => x / expSum);

        return {
            hiddenLayer,
            actionProbabilities
        };
    }

    // Get the best action based on the current state
    getAction(state, currentDirection, checkCollisionFn) {
        const { hiddenLayer, actionProbabilities } = this.forward(state, currentDirection);
        const actions = ['up', 'right', 'down', 'left'];

        // Apply epsilon-greedy exploration
        let chosenAction;
        if (Math.random() < this.explorationRate) {
            // Explore: choose a random action
            const randomIndex = Math.floor(Math.random() * actions.length);
            chosenAction = actions[randomIndex];
        } else {
            // Exploit: choose the best action according to the model
            let maxIndex = 0;
            for (let i = 1; i < actionProbabilities.length; i++) {
                if (actionProbabilities[i] > actionProbabilities[maxIndex]) {
                    maxIndex = i;
                }
            }
            chosenAction = actions[maxIndex];
        }

        // Check if the chosen action is valid (doesn't cause immediate collision)
        const directions = {
            up: { x: 0, y: -1 },
            right: { x: 1, y: 0 },
            down: { x: 0, y: 1 },
            left: { x: -1, y: 0 }
        };

        // Prevent 180-degree turns
        const oppositeDirections = {
            up: 'down',
            right: 'left',
            down: 'up',
            left: 'right'
        };

        if (chosenAction === oppositeDirections[currentDirection]) {
            // If trying to go in the opposite direction, find another action
            const validActions = actions.filter(action =>
                action !== oppositeDirections[currentDirection] &&
                !checkCollisionFn({
                    x: state.headX + directions[action].x,
                    y: state.headY + directions[action].y
                })
            );

            if (validActions.length > 0) {
                // Choose the action with highest probability among valid actions
                let bestValidAction = validActions[0];
                let bestValidProb = actionProbabilities[actions.indexOf(validActions[0])];

                for (let i = 1; i < validActions.length; i++) {
                    const actionIndex = actions.indexOf(validActions[i]);
                    if (actionProbabilities[actionIndex] > bestValidProb) {
                        bestValidProb = actionProbabilities[actionIndex];
                        bestValidAction = validActions[i];
                    }
                }

                chosenAction = bestValidAction;
            }
        }

        // Learn from this decision
        this.learn(state, chosenAction, hiddenLayer);

        return chosenAction;
    }

    // Store experience for training
    storeExperience(state, action, reward, nextState, done) {
        // Convert action to index
        const actionIndex = ['up', 'right', 'down', 'left'].indexOf(action);

        // Store experience in replay memory
        this.replayMemory.push({
            state: { ...state }, // Clone state to avoid reference issues
            actionIndex,
            reward,
            nextState: nextState ? { ...nextState } : null,
            done
        });

        // Limit memory size
        if (this.replayMemory.length > this.maxMemorySize) {
            this.replayMemory.shift(); // Remove oldest experience
        }
    }

    // Enhanced learning with experience replay
    learn(state, chosenAction, hiddenLayer) {
        // Store current state and action for later training
        this.lastState = { ...state };
        this.lastAction = chosenAction;
        this.lastHiddenLayer = [...hiddenLayer];

        // Calculate immediate reward based on distance to food
        const distanceToFood = Math.abs(state.headX - state.foodX) + Math.abs(state.headY - state.foodY);
        const maxDistance = this.gridWidth + this.gridHeight;
        const reward = 1 - (distanceToFood / maxDistance); // Higher reward when closer to food

        // Update weights with immediate reward
        const actionIndex = ['up', 'right', 'down', 'left'].indexOf(chosenAction);

        // Update output layer weights for the chosen action
        for (let i = 0; i < hiddenLayer.length; i++) {
            this.hiddenToOutput[i][actionIndex] +=
                this.learningRate * reward * hiddenLayer[i];
        }

        // Update output bias
        this.outputBias[actionIndex] += this.learningRate * reward;

        // Train on batch from replay memory if we have enough samples
        if (this.replayMemory.length >= this.batchSize) {
            this.trainOnBatch();
        }
    }

    // Train on a batch of experiences from replay memory
    trainOnBatch() {
        // Skip if not enough samples
        if (this.replayMemory.length < this.batchSize) {
            return;
        }

        // Sample a batch of experiences
        const batch = [];
        for (let i = 0; i < this.batchSize; i++) {
            const randomIndex = Math.floor(Math.random() * this.replayMemory.length);
            batch.push(this.replayMemory[randomIndex]);
        }

        let totalLoss = 0;

        // Train on each experience in the batch
        for (const experience of batch) {
            const { state, actionIndex, reward, nextState, done } = experience;

            // Skip if state or nextState is null
            if (!state || !nextState) continue;

            // Forward pass for current state
            const currentDirection =
                state.currentDirection ||
                (state.directionUp ? 'up' :
                 state.directionRight ? 'right' :
                 state.directionDown ? 'down' : 'left');

            const { actionProbabilities } = this.forward(state, currentDirection);

            // Forward pass for next state to get target values
            const nextDirection =
                nextState.currentDirection ||
                (nextState.directionUp ? 'up' :
                 nextState.directionRight ? 'right' :
                 nextState.directionDown ? 'down' : 'left');

            const { actionProbabilities: nextActionProbs } = this.forward(nextState, nextDirection);

            // Calculate target Q-value using Q-learning update rule
            const maxNextQ = Math.max(...nextActionProbs);
            const targetQ = done ? reward : reward + 0.95 * maxNextQ; // 0.95 is discount factor

            // Calculate loss (squared error)
            const currentQ = actionProbabilities[actionIndex];
            const error = targetQ - currentQ;
            const loss = error * error;
            totalLoss += loss;

            // Update weights for the action taken
            const { hiddenLayer } = this.forward(state, currentDirection);

            // Update output layer weights
            for (let i = 0; i < hiddenLayer.length; i++) {
                this.hiddenToOutput[i][actionIndex] +=
                    this.learningRate * error * hiddenLayer[i];
            }

            // Update output bias
            this.outputBias[actionIndex] += this.learningRate * error;
        }

        // Update training statistics
        this.trainingCount++;
        this.totalLoss += totalLoss / this.batchSize;
        this.averageLoss = this.totalLoss / this.trainingCount;

        // Gradually reduce exploration rate as training progresses
        this.explorationRate = Math.max(0.05, this.explorationRate * 0.999);

        // Log training progress occasionally
        if (this.trainingCount % 10 === 0) {
            console.log(`Training batch #${this.trainingCount}, Avg Loss: ${this.averageLoss.toFixed(4)}, Exploration: ${this.explorationRate.toFixed(4)}`);
        }
    }

    // Handle food eaten event - give high reward
    handleFoodEaten(state) {
        if (this.lastState && this.lastAction) {
            // Give a high reward for eating food
            this.storeExperience(
                this.lastState,
                this.lastAction,
                10.0, // High reward for eating food
                state,
                false // Not done yet
            );

            console.log('Food eaten! Stored positive experience with reward: 10.0');

            // Train immediately after eating food
            if (this.replayMemory.length >= this.batchSize) {
                this.trainOnBatch();
            }

            // Save model periodically after eating food
            if (Math.random() < 0.2) { // 20% chance to save after eating
                this.saveModel();
            }
        }
    }

    // Handle game over event - give negative reward
    handleGameOver(state) {
        if (this.lastState && this.lastAction) {
            // Give a negative reward for game over
            this.storeExperience(
                this.lastState,
                this.lastAction,
                -10.0, // Negative reward for dying
                state,
                true // Game is done
            );

            console.log('Game over! Stored negative experience with reward: -10.0');

            // Train immediately after game over
            if (this.replayMemory.length >= this.batchSize) {
                this.trainOnBatch();
            }

            // Always save model after game over
            this.saveModel();
        }
    }

    // Get training statistics
    getTrainingStats() {
        return {
            memorySize: this.replayMemory.length,
            batchSize: this.batchSize,
            trainingCount: this.trainingCount,
            averageLoss: this.averageLoss,
            explorationRate: this.explorationRate
        };
    }

    // Save the model weights to localStorage
    saveModel() {
        try {
            const modelData = {
                inputToHidden: this.inputToHidden,
                hiddenToOutput: this.hiddenToOutput,
                hiddenBias: this.hiddenBias,
                outputBias: this.outputBias,
                trainingCount: this.trainingCount,
                totalLoss: this.totalLoss,
                averageLoss: this.averageLoss,
                explorationRate: this.explorationRate,
                replayMemorySize: this.replayMemory.length
            };

            localStorage.setItem('simplified-deep-ai-model', JSON.stringify(modelData));
            console.log(`Saved simplified deep AI model to localStorage (Training: ${this.trainingCount}, Memory: ${this.replayMemory.length})`);
            return true;
        } catch (error) {
            console.error('Failed to save simplified deep AI model:', error);
            return false;
        }
    }

    // Load the model weights from localStorage
    loadModel() {
        try {
            const modelData = JSON.parse(localStorage.getItem('simplified-deep-ai-model'));
            if (modelData) {
                this.inputToHidden = modelData.inputToHidden;
                this.hiddenToOutput = modelData.hiddenToOutput;
                this.hiddenBias = modelData.hiddenBias;
                this.outputBias = modelData.outputBias;

                // Load training statistics if available
                if (modelData.trainingCount !== undefined) {
                    this.trainingCount = modelData.trainingCount;
                }
                if (modelData.totalLoss !== undefined) {
                    this.totalLoss = modelData.totalLoss;
                }
                if (modelData.averageLoss !== undefined) {
                    this.averageLoss = modelData.averageLoss;
                }
                if (modelData.explorationRate !== undefined) {
                    this.explorationRate = modelData.explorationRate;
                }

                console.log(`Loaded simplified deep AI model from localStorage (Training: ${this.trainingCount}, Exploration: ${this.explorationRate.toFixed(4)})`);
                return true;
            }
            return false;
        } catch (error) {
            console.error('Failed to load simplified deep AI model:', error);
            return false;
        }
    }
}
