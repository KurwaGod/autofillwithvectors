using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

namespace BashAutocomplete
{
    class Program
    {
        static void Main(string[] args)
        {
            // Initialize the neural network
            var autocompleteNN = new AutocompleteNeuralNetwork();
            
            // Load previously saved model if it exists
            string modelPath = "autocomplete_model.dat";
            if (File.Exists(modelPath))
            {
                autocompleteNN.LoadModel(modelPath);
                Console.WriteLine("Loaded existing model.");
            }
            
            // History will store all previously entered commands
            var history = new List<string>();
            
            // Main input loop
            Console.WriteLine("Bash Shell with Neural Network Autocomplete");
            Console.WriteLine("Type 'exit' to quit, 'train' to retrain the model");
            
            while (true)
            {
                Console.Write("$ ");
                StringBuilder inputBuilder = new StringBuilder();
                ConsoleKeyInfo keyInfo;
                string suggestion = "";
                
                // Read input character by character to enable autocompletion
                while ((keyInfo = Console.ReadKey(true)).Key != ConsoleKey.Enter)
                {
                    if (keyInfo.Key == ConsoleKey.Tab)
                    {
                        // Accept the suggestion
                        if (!string.IsNullOrEmpty(suggestion))
                        {
                            inputBuilder.Append(suggestion);
                            Console.Write(suggestion);
                            suggestion = "";
                        }
                        continue;
                    }
                    
                    if (keyInfo.Key == ConsoleKey.Backspace && inputBuilder.Length > 0)
                    {
                        inputBuilder.Length--;
                        Console.Write("\b \b");
                        suggestion = "";
                    }
                    else if (keyInfo.KeyChar >= 32 && keyInfo.KeyChar <= 126) // Printable chars
                    {
                        inputBuilder.Append(keyInfo.KeyChar);
                        Console.Write(keyInfo.KeyChar);
                    }
                    
                    // Get autocomplete suggestion if we have at least 2 characters
                    if (inputBuilder.Length >= 2)
                    {
                        string input = inputBuilder.ToString();
                        suggestion = autocompleteNN.GetSuggestion(input);
                        
                        if (!string.IsNullOrEmpty(suggestion))
                        {
                            Console.ForegroundColor = ConsoleColor.DarkGray;
                            Console.Write(suggestion);
                            Console.ForegroundColor = ConsoleColor.Gray;
                            Console.CursorLeft = input.Length + 2; // +2 for the "$ " prompt
                        }
                    }
                }
                
                Console.WriteLine(); // Move to the next line after Enter
                
                string input = inputBuilder.ToString();
                
                if (input.ToLower() == "exit")
                {
                    break;
                }
                else if (input.ToLower() == "train")
                {
                    Console.WriteLine("Training the model with current history...");
                    autocompleteNN.Train(history);
                    autocompleteNN.SaveModel(modelPath);
                    Console.WriteLine("Training complete and model saved.");
                }
                else
                {
                    // Add to history and train the network with this new example
                    history.Add(input);
                    
                    // Simulate command execution
                    Console.WriteLine($"Executing: {input}");
                    
                    // Every 5 commands, update the model
                    if (history.Count % 5 == 0)
                    {
                        autocompleteNN.Train(history);
                        autocompleteNN.SaveModel(modelPath);
                    }
                }
            }
        }
    }
    
    public class AutocompleteNeuralNetwork
    {
        // Simple vector representation of words
        private Dictionary<string, Dictionary<string, int>> wordFollowFrequency;
        private Dictionary<string, List<float>> wordVectors;
        private int vectorSize = 10;
        private Random random = new Random();
        
        public AutocompleteNeuralNetwork()
        {
            wordFollowFrequency = new Dictionary<string, Dictionary<string, int>>();
            wordVectors = new Dictionary<string, List<float>>();
        }
        
        // Generate a random vector for a new word
        private List<float> GenerateRandomVector()
        {
            var vector = new List<float>();
            for (int i = 0; i < vectorSize; i++)
            {
                vector.Add((float)random.NextDouble() * 2 - 1); // Between -1 and 1
            }
            return vector;
        }
        
        // Compute similarity between two vectors (cosine similarity)
        private float ComputeSimilarity(List<float> v1, List<float> v2)
        {
            float dotProduct = 0;
            float mag1 = 0;
            float mag2 = 0;
            
            for (int i = 0; i < vectorSize; i++)
            {
                dotProduct += v1[i] * v2[i];
                mag1 += v1[i] * v1[i];
                mag2 += v2[i] * v2[i];
            }
            
            mag1 = (float)Math.Sqrt(mag1);
            mag2 = (float)Math.Sqrt(mag2);
            
            if (mag1 == 0 || mag2 == 0)
                return 0;
                
            return dotProduct / (mag1 * mag2);
        }
        
        // Train the network with a list of commands
        public void Train(List<string> history)
        {
            // Count word pair frequencies
            foreach (var command in history)
            {
                string[] words = command.Split(' ');
                
                for (int i = 0; i < words.Length - 1; i++)
                {
                    string currentWord = words[i];
                    string nextWord = words[i + 1];
                    
                    // Add vectors for words we haven't seen before
                    if (!wordVectors.ContainsKey(currentWord))
                    {
                        wordVectors[currentWord] = GenerateRandomVector();
                    }
                    
                    if (!wordVectors.ContainsKey(nextWord))
                    {
                        wordVectors[nextWord] = GenerateRandomVector();
                    }
                    
                    // Update frequency counts
                    if (!wordFollowFrequency.ContainsKey(currentWord))
                    {
                        wordFollowFrequency[currentWord] = new Dictionary<string, int>();
                    }
                    
                    if (!wordFollowFrequency[currentWord].ContainsKey(nextWord))
                    {
                        wordFollowFrequency[currentWord][nextWord] = 0;
                    }
                    
                    wordFollowFrequency[currentWord][nextWord]++;
                    
                    // Update vectors to be closer for words that occur together
                    UpdateVectors(currentWord, nextWord);
                }
            }
        }
        
        // Update vectors for words that occur together
        private void UpdateVectors(string word1, string word2)
        {
            float learningRate = 0.1f;
            
            var v1 = wordVectors[word1];
            var v2 = wordVectors[word2];
            
            // Move vectors closer
            for (int i = 0; i < vectorSize; i++)
            {
                float adjustment = (v2[i] - v1[i]) * learningRate;
                v1[i] += adjustment;
                v2[i] -= adjustment;
            }
            
            // Normalize vectors
            NormalizeVector(v1);
            NormalizeVector(v2);
        }
        
        private void NormalizeVector(List<float> vector)
        {
            float magnitude = 0;
            for (int i = 0; i < vector.Count; i++)
            {
                magnitude += vector[i] * vector[i];
            }
            
            magnitude = (float)Math.Sqrt(magnitude);
            
            if (magnitude > 0)
            {
                for (int i = 0; i < vector.Count; i++)
                {
                    vector[i] /= magnitude;
                }
            }
        }
        
        // Get suggestion for partial input
        public string GetSuggestion(string partialInput)
        {
            // Check for exact matches first (current word followed by most common next word)
            string lastWord = partialInput.Split(' ').Last();
            string beforeLastWords = partialInput.Substring(0, partialInput.Length - lastWord.Length);
            
            // If the last word already has a space, we need to suggest a new word
            if (partialInput.EndsWith(" "))
            {
                lastWord = "";
                beforeLastWords = partialInput;
            }
            
            // If there's a previous word that we can use for prediction
            string previousWord = beforeLastWords.Trim().Split(' ').LastOrDefault();
            
            if (!string.IsNullOrEmpty(previousWord) && wordFollowFrequency.ContainsKey(previousWord))
            {
                // Find all words that start with lastWord
                var candidates = wordFollowFrequency[previousWord]
                    .Where(x => x.Key.StartsWith(lastWord))
                    .OrderByDescending(x => x.Value)
                    .ToList();
                
                if (candidates.Count > 0)
                {
                    string bestMatch = candidates.First().Key;
                    return bestMatch.Substring(lastWord.Length);
                }
            }
            
            // If no exact match, use vector similarity to find the closest match
            return GetSuggestionByVectorSimilarity(partialInput);
        }
        
        private string GetSuggestionByVectorSimilarity(string partialInput)
        {
            // Split the input to get the last word
            string[] words = partialInput.Split(' ');
            string lastWord = words.Last();
            
            // If last word is empty (e.g., "command " with trailing space)
            if (string.IsNullOrEmpty(lastWord))
            {
                string previousWord = words.Length > 1 ? words[words.Length - 2] : "";
                
                if (!string.IsNullOrEmpty(previousWord) && wordFollowFrequency.ContainsKey(previousWord))
                {
                    // Find the most common word following the previous word
                    var mostCommon = wordFollowFrequency[previousWord].OrderByDescending(x => x.Value).First();
                    return mostCommon.Key;
                }
                
                return "";
            }
            
            // Find partially matching words
            var matchingWords = wordVectors.Keys
                .Where(w => w.StartsWith(lastWord) && w != lastWord)
                .ToList();
            
            if (matchingWords.Count == 0)
                return "";
            
            // With only one matching word, return it
            if (matchingWords.Count == 1)
                return matchingWords[0].Substring(lastWord.Length);
            
            // For multiple matches, use vector similarity to find the best one
            if (words.Length > 1)
            {
                string previousWord = words[words.Length - 2];
                
                if (wordVectors.ContainsKey(previousWord))
                {
                    var prevVector = wordVectors[previousWord];
                    
                    // Find the most similar word among matches based on vector similarity
                    float bestSimilarity = -1;
                    string bestMatch = "";
                    
                    foreach (var match in matchingWords)
                    {
                        if (wordVectors.ContainsKey(match))
                        {
                            float similarity = ComputeSimilarity(prevVector, wordVectors[match]);
                            if (similarity > bestSimilarity)
                            {
                                bestSimilarity = similarity;
                                bestMatch = match;
                            }
                        }
                    }
                    
                    if (!string.IsNullOrEmpty(bestMatch))
                    {
                        return bestMatch.Substring(lastWord.Length);
                    }
                }
            }
            
            // Default: return the first match
            return matchingWords[0].Substring(lastWord.Length);
        }
        
        // Save the model to a file
        public void SaveModel(string path)
        {
            using (var writer = new StreamWriter(path))
            {
                // Save word vectors
                writer.WriteLine(wordVectors.Count);
                foreach (var pair in wordVectors)
                {
                    writer.WriteLine(pair.Key);
                    writer.WriteLine(string.Join(",", pair.Value));
                }
                
                // Save word frequencies
                writer.WriteLine(wordFollowFrequency.Count);
                foreach (var word in wordFollowFrequency)
                {
                    writer.WriteLine(word.Key);
                    writer.WriteLine(word.Value.Count);
                    
                    foreach (var followPair in word.Value)
                    {
                        writer.WriteLine($"{followPair.Key},{followPair.Value}");
                    }
                }
            }
        }
        
        // Load the model from a file
        public void LoadModel(string path)
        {
            using (var reader = new StreamReader(path))
            {
                // Load word vectors
                int vectorCount = int.Parse(reader.ReadLine());
                for (int i = 0; i < vectorCount; i++)
                {
                    string word = reader.ReadLine();
                    string[] components = reader.ReadLine().Split(',');
                    var vector = components.Select(c => float.Parse(c)).ToList();
                    wordVectors[word] = vector;
                }
                
                // Load word frequencies
                int freqCount = int.Parse(reader.ReadLine());
                for (int i = 0; i < freqCount; i++)
                {
                    string word = reader.ReadLine();
                    int followCount = int.Parse(reader.ReadLine());
                    
                    var followDict = new Dictionary<string, int>();
                    for (int j = 0; j < followCount; j++)
                    {
                        string[] parts = reader.ReadLine().Split(',');
                        followDict[parts[0]] = int.Parse(parts[1]);
                    }
                    
                    wordFollowFrequency[word] = followDict;
                }
            }
        }
    }
}
