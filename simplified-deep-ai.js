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
    
    // Simple learning: adjust weights based on reward
    learn(state, chosenAction, hiddenLayer) {
        if (Math.random() < 0.1) { // Only update weights occasionally
            // Calculate reward based on distance to food
            const distanceToFood = Math.abs(state.headX - state.foodX) + Math.abs(state.headY - state.foodY);
            const maxDistance = this.gridWidth + this.gridHeight;
            const reward = 1 - (distanceToFood / maxDistance); // Higher reward when closer to food
            
            // Update weights with a small learning rate
            const actionIndex = ['up', 'right', 'down', 'left'].indexOf(chosenAction);
            
            // Update output layer weights for the chosen action
            for (let i = 0; i < hiddenLayer.length; i++) {
                this.hiddenToOutput[i][actionIndex] += 
                    this.learningRate * reward * hiddenLayer[i];
            }
            
            // Update output bias
            this.outputBias[actionIndex] += this.learningRate * reward;
        }
    }
    
    // Save the model weights to localStorage
    saveModel() {
        try {
            const modelData = {
                inputToHidden: this.inputToHidden,
                hiddenToOutput: this.hiddenToOutput,
                hiddenBias: this.hiddenBias,
                outputBias: this.outputBias
            };
            
            localStorage.setItem('simplified-deep-ai-model', JSON.stringify(modelData));
            console.log('Saved simplified deep AI model to localStorage');
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
                console.log('Loaded simplified deep AI model from localStorage');
                return true;
            }
            return false;
        } catch (error) {
            console.error('Failed to load simplified deep AI model:', error);
            return false;
        }
    }
}
