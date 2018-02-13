import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

public class Exercise2 {
	
	// What model to use to train the algorithm; values: 
	// Bernoulli
	// Multinomial
	private static  String modelType = "Bernoulli";
	// Cross validation with kfold
	private static  String validation = "kfold";
	private static  int k = 10;
	// set all text to lower case to ignore upper cases
	private static  boolean setLowerCase = false;
	// remove dots and commas from the text and replace them with a space to avoid merging words
	private static  boolean removeSpecialChar = false;
	// LIST OF CLASSES
	private static Set<String> classList = new HashSet<String>();
	//private static List<String> classList = Arrays.asList("malware","safe");
	
	
	// Global variables UPDATED ON EACH ITERATION
	// List of classes (key) with a list of documents in that class
	private static Map<String, List<String>> myDictionary = new HashMap<String, List<String>>();
	// List of unique words, for each we have a map of cond prob for each class
	private static Map<String, Map<String, Double>> condProbab = new HashMap<String, Map<String, Double>>();
	// Number of documents per class
	private static Map<String, Integer> linesPerClass= new HashMap<String, Integer>();
	// Prior probability for each class
	private static Map<String, Double> priorProbClass= new HashMap<String, Double>();
	// List of unique words in the dataset
	private static HashSet<String> vocabulary = new HashSet<String>();
	
	// Vocabulary per class - For multinomial model
	private static Map<String, List<String>> classVocabulary = new HashMap<String, List<String>>();

	
	public static void main(String[] args) {
	/*	if (args.length > 0 ){
			for (int a=0; a<args.length; a++){
				switch(a){
					case 0:
						if (args[0].equals("0")){
							setLowerCase = false;
						} else if (args[0].equalsIgnoreCase("help")){
							printHelp();	
						}
				}
				
			}
		}*/
		
		
		// Get SMSs
		//List<String> myHam = new ArrayList<>(readLines("/Users/JoseMa/SMSSpamCollection", "ham"));
		//List<String> mySpam = new ArrayList<>(readLines("/Users/JoseMa/SMSSpamCollection", "spam"));
		//List<String> fullFile = new ArrayList<>(readLines("/Users/JoseMa/Dropbox/Ms in AI and Robotics/Machine Learning/Exercises/Exercise 2/smsspamcollection/SMSSpamCollection"));
		List<String> fullFile = new ArrayList<>(readLines("/Users/JoseMa/malwareDataFile_Bayes_1510503224346_m_cp_1_150.csv"));

		getClasses(fullFile);
		
		if (validation.equals("kfold")){
			int bags = Math.round(fullFile.size()/k);
			println("****************");
			println("* Starting a k-fold cross validation with: ");
			println("*   - Size of the dataset: " + fullFile.size());
			println("*   - Number of bags: " + k);
			println("*   - Bag size: " + bags);
			println("****************");
			String resultClass = "";
			String actualClass = "";
			int rightClasses = 0;
			int wrongClasses = 0;
			int totalTests = 0;
			Map<String,Map<String,Integer>> confusionMatrix = new HashMap <String,Map<String,Integer>>();
			for(String myClass : classList){
				Map<String,Integer> classCounter = new HashMap<String,Integer>();
				confusionMatrix.put(myClass, classCounter);
			}
			
			for(int b=1; b<=k; b++){
				println("****************");
				println(" Iteration - " + b);
				println("****************");
				Map<Integer,List<String>> myLists = docSplit(b, bags, fullFile);
				println("Training set size: " + myLists.get(1).size());
				println("Validation set size: " + myLists.get(0).size());			
				
				if(modelType.equals("Multinomial")){
					trainMultinomialModel(classList, myLists.get(1));
				} else {
					trainBernoulliModel(classList, myLists.get(1));
				}
				
				List<String> validationList = myLists.get(0);
				Integer numberResults=0;
				for (int l=0; l<validationList.size(); l++){
					String nextDoc = validationList.get(l);
					actualClass = nextDoc.substring(0, nextDoc.indexOf("\t"));
					if(modelType.equals("Multinomial")){
						resultClass = applyMultinomialModel(classList, nextDoc.substring(nextDoc.indexOf("\t")+1,nextDoc.length()));
					} else {
						resultClass = applyBernoulliModel(classList, nextDoc.substring(nextDoc.indexOf("\t")+1,nextDoc.length()));
					}
					resultClass = applyBernoulliModel(classList, nextDoc.substring(nextDoc.indexOf("\t")+1,nextDoc.length()));
					/*println("********");
					println("MSG: " + nextDoc);
					println("Expected: " + actualClass);
					println("Result: " + resultClass);
					println("********");*/
					totalTests++;
					if(resultClass.equals(actualClass)){
						rightClasses++;
					} else {
						wrongClasses++;
					}
					Map<String,Integer> classCounter;
					if(confusionMatrix.get(actualClass).isEmpty()){
						classCounter = new HashMap<String,Integer>();
						classCounter.put(resultClass, 1);
					} else {
						
						classCounter = new HashMap<String,Integer>(confusionMatrix.get(actualClass));
						if(classCounter.containsKey(resultClass)){
							numberResults =  classCounter.get(resultClass);
							numberResults+=1;
							classCounter.replace(resultClass, numberResults);
						} else {
							classCounter = new HashMap<String,Integer>(classCounter);
							classCounter.put(resultClass, 1);
						}
						
					}
					
					confusionMatrix.replace(actualClass, classCounter);
				}		
			}
			println("*************************");
			println("* RESULTS OF ALGORITHM");
			println("* Total number of tests: " + totalTests);
			println("* Right classes: " + rightClasses);
			println("* Wrong classes: " + wrongClasses);
			double accuracy = ((rightClasses+0.0)/(totalTests+0.0))*100;
			println("* % of right classifications: " + accuracy +"%");
			println("*************************");
			println("");
			println("*************************");
			println("* Confusion Matrix");
			
			Map<String,String> legend = new HashMap<String,String>();
			String columns ="";
			Iterator <String> classIterator = classList.iterator();
			int l=0;
			while(classIterator.hasNext()){
				l++;
				columns = columns+"["+l+"]"+"\t";
				legend.put(classIterator.next(),"["+l+"]");
			}
			println(columns);
			int totalCount = 0;
			for (String myClass : classList){
				String newLine = "";
				Map<String,Integer> classCounter = 
						new HashMap<String,Integer>(confusionMatrix.get(myClass));
				for (String myClassR : classList){
					if (classCounter.containsKey(myClassR)){
						totalCount = classCounter.get(myClassR);
					}else {
						totalCount=0;
					}
					newLine = newLine + totalCount + "\t";
				}
				println(newLine + " | " + legend.get(myClass)  + " -> " + myClass);
			}
			//println("LEGEND:");
			//println(legend.toString());
			
			

		}
	}
		
	/********
	 * 
	 * @param classList: List of strings containing the list of classes
	 * @param docList: List of strings containing the dataset to train the algorithm
	 * 
	 * This function calls the required functions to generate the data for the algorithm: 
	 *  - Dictionary: documents classified by class
	 *  - Vocabulary: list of unique words in the training dataset
	 *  - Conditional Probability: for each word in each class. 
	 * 
	 */
	private static void trainBernoulliModel(Set<String> classList2, List<String> docList){
		// Create my dictionary class Vs SMS
		// and initialise variables
		generateDictionary(classList2, docList);
		
		//Generate Vocabulary = unique list of words among all documents in the dataset
		generateVocabulary(docList);
		
		//Calculate conditional probability for each term in a class
		calculateCondProbBernoulli();
		
	}
	
	private static void trainMultinomialModel(Set<String> classList2, List<String> docList){
		// Create my dictionary class Vs SMS
		// and initialise variables
		generateDictionary(classList2, docList);
		
		//Generate Vocabulary = unique list of words among all documents in the dataset
		generateVocabulary(docList);
		
		//Generate Vocabulary = all words among all documents in the dataset classified by class
		generateClassVocabulary();
		
		//Calculate conditional probability for each term in a class
		calculateCondProbMultinomial();
		
	}
	
	/*********
	 * 
	 * @param classList: List of strings containing the list of classes
	 * @param newDoc: String containing the new document to test
	 * @return: the class calculated for that document
	 */
	private static String applyBernoulliModel(Set<String> classList2, String newDoc){
		//Extract tokens from document
		String tokens[] = newDoc.split(" ");
		
		//Iterator<String> myWordsIterator = newDoc.split(" ");
		String myClass;
		Map<String, Double> classScore = new HashMap<String,Double>();
		double score = 0.0;
		double priorProbC = 0.0;
		//writeOutput(condProbab.toString().replace("={", "\",").replaceAll("}, ", "\n"));
		//writeOutput(condProbab.toString());
		//println("NEW DOC: " + newDoc);
		for (String c : classList2){
			score=0.0;
			Iterator<String> myVocabIterator = vocabulary.iterator();
			myClass = c;
			priorProbC = priorProbClass.get(myClass);
			score = priorProbC;
			//println("Prior prob for class " + myClass +": "+priorProbC);
			while(myVocabIterator.hasNext()){
				String myWord = myVocabIterator.next();
				double condProbWordClass = condProbab.get(myWord).get(myClass);
				//if (condProbWordClass == 0.0) println("Word " +myWord + " has prob "+condProbWordClass);
				//println("Conditional Prob for class " + myClass +" and word "+myWord+": "+condProbWordClass);
				if(newDoc.contains(myWord)){
					
					//score += Math.log(condProbWordClass);
					score *= condProbWordClass;
				} else {

					//score += Math.log(1-condProbWordClass);
					score *= (1.0-condProbWordClass);
				}
				
			}
			
			// Add the score for the class we just checked
			classScore.put(myClass, score);
			
		}

		// Find the class argmax of score
		Iterator<String> scoreClassIt = classScore.keySet().iterator();
		//println(classScore.toString());
		String myScoreClass ="";
		String argmaxClass = "";
		// Initialising to a very negative number in case log is used to calculate the classification
		// Log of a value less than 1 is negative.
		Double argmaxScore = -100000000000.0;
		while (scoreClassIt.hasNext()){
			myScoreClass = scoreClassIt.next();
			
			if (argmaxScore<classScore.get(myScoreClass)){
				argmaxClass = myScoreClass;
				argmaxScore = classScore.get(myScoreClass);
			}
		}
		return argmaxClass;
		//return "ham";
	}
	
	
	
	/*********
	 * 
	 * @param classList: List of strings containing the list of classes
	 * @param newDoc: String containing the new document to test
	 * @return: the class calculated for that document
	 */
	private static String applyMultinomialModel(Set<String> classList2, String newDoc){
		//Extract tokens from document
		String tokens[] = newDoc.split(" ");

		String myClass;
		Map<String, Double> classScore = new HashMap<String,Double>();
		double score = 0.0;
		double priorProbC = 0.0;
		// Iterate through the classes
		for (String c : classList2){
			score=0.0;
			myClass = c;
			// Get prior probability for this class
			priorProbC = priorProbClass.get(myClass);
			//score = Math.log(priorProbC);
			score = priorProbC;
			
			//For each token in the document, check if there is conditional probability for this class
			//If there isn't set it to 0;
			for(int t=0; t<tokens.length;t++){
				String myWord = tokens[t];
				
				double condProbWordClass = 0.0;
				if (condProbab.containsKey(myWord)){
					condProbWordClass=condProbab.get(myWord).get(myClass);
				}
				
				//score += Math.log(condProbWordClass);
				score *= condProbWordClass;
				
			}
			
			// Add the score for the class we just checked
			classScore.put(myClass, score);
			
		}

		// Find the class argmax of score
		Iterator<String> scoreClassIt = classScore.keySet().iterator();
		//println(classScore.toString());
		String myScoreClass ="";
		String argmaxClass = "";
		// Initialising to a very negative number in case log is used to calculate the classification
		// Log of a value less than 1 is negative.
		Double argmaxScore = -100000000000.0;
		while (scoreClassIt.hasNext()){
			myScoreClass = scoreClassIt.next();
			
			if (argmaxScore<classScore.get(myScoreClass)){
				argmaxClass = myScoreClass;
				argmaxScore = classScore.get(myScoreClass);
			}
		}
		return argmaxClass;
		//return "ham";
	}
	
	/********
	 * 
	 * @param classList: list of strings containing the list of classes 
	 * @param docList: list of strings containing all the documents
	 * 
	 * This function classifies all the documents by class.
	 * It populates the global hashmap variable 'myDictionary' where:
	 *   - Key = class
	 *   - Value = List of strings containing the documents
	 *   
	 * It also populates two other global variables:
	 *   - priorProbClass: Map string-double containing the prior probability per class
	 *   - linesPerClass: Map string-integer containing the number of documents per class
	 */
	private static void generateDictionary(Set<String> classList2, List<String> docList){
		
		myDictionary.clear();
		//String classifLines = new String[];
		
		//Generate my dictionary class - document
		
		Iterator<String> myClasses = classList2.iterator();
		
		while (myClasses.hasNext()){
			final String myClass = myClasses.next();
			List<String> classifiedLines = new ArrayList<>();
			docList.stream().filter(c -> c.startsWith(myClass))
							.forEach(s -> classifiedLines.add(s.substring(s.indexOf("\t") +1)));
			myDictionary.put(myClass, classifiedLines);
			//Add prior probability for this class
			priorProbClass.put(myClass, ((double)classifiedLines.size()/(double)docList.size()));
		}
		
		//Count number of documents per class and put it in a map 
		myDictionary.keySet().forEach(s -> linesPerClass.put(s, myDictionary.get(s).size()));
		
		println("Dictionary generated. Number of docs per class: ");
		println(linesPerClass.toString());
		
	}
	
	/*********
	 * Class that populates the global variable 'classVocabulary' using the classified documents variable
	 * 	This vocabulary includes all the occurrences of each word.
	 */
	private static void generateClassVocabulary(){
			
		classVocabulary.clear();
	
			
		//Generate my dictionary class - document
		
		Iterator<String> classIterator = myDictionary.keySet().iterator();
		
		while (classIterator.hasNext()){
			final String myClass = classIterator.next();
			List<String> classifiedLines = new ArrayList<>(myDictionary.get(myClass));
			List<String> vocab = new ArrayList<String>();
			String token[];
			for(int d=0; d<classifiedLines.size();d++){
				token = classifiedLines.get(d).split(" ");
				for(int t=0;t<token.length; t++){
					vocab.add(token[t]);
				}		
			}
			classVocabulary.put(myClass, vocab);
			
		}
		
		//Count number of documents per class and put it in a map 
		myDictionary.keySet().forEach(s -> linesPerClass.put(s, myDictionary.get(s).size()));
		
		println("Classes vocabulary generated. Number of tokens per class: ");
		println(linesPerClass.toString());
			
	}
	
	
	/***********
	 * 
	 * @param docList: a list of strings containing all the documents
	 * This function populates the global variable 'vocabulary' with a list of unique words in the dataset
	 * there is an extra global variable 'removeSpecialChar' to remove the '.' character
	 * 
	 */
	private static void generateVocabulary(List<String> docList){
		Iterator<String> myDocsIterator = docList.iterator();
		String docWords[];
		String doc = "";
		while (myDocsIterator.hasNext()){
			doc = myDocsIterator.next();
			if (removeSpecialChar){
				docWords = doc.substring(doc.indexOf("\t")).toString().replace(".", " ").replace(",", " ").trim().toString().split(" ");
			} else {
				docWords = doc.substring(doc.indexOf("\t")).toString().toString().split(" ");	
			}
			for (int i=0; i< docWords.length; i++){
				vocabulary.add(docWords[i].trim());
			}
		}
		vocabulary.remove("");
		vocabulary.remove(" ");
		
		println("Vocabulary generated - number of unique words: " + vocabulary.size());
	}
	
	/***********
	 * 
	 * @param docList: a list of strings containing all the documents
	 * This function populates the global variable 'vocabularyPerClass' with a list of all words in the dataset classified
	 * there is an extra global variable 'removeSpecialChar' to remove the '.' character
	 * 
	 */
	private static void generateVocabularyPerClass(List<String> docList){
		Iterator<String> myDocsIterator = docList.iterator();
		String docWords[];
		String doc = "";
		while (myDocsIterator.hasNext()){
			doc = myDocsIterator.next();
			if (removeSpecialChar){
				docWords = doc.substring(doc.indexOf("\t")).toString().replace(".", " ").replace(",", " ").trim().toString().split(" ");
			} else {
				docWords = doc.substring(doc.indexOf("\t")).toString().toString().split(" ");	
			}
			for (int i=0; i< docWords.length; i++){
				vocabulary.add(docWords[i].trim());
			}
		}
		vocabulary.remove("");
		vocabulary.remove(" ");
		
		println("Vocabulary generated - number of unique words: " + vocabulary.size());
	}
	
	private static void calculateCondProbBernoulli() {
		println (" - Calculating Bernoulli cond prob with vocabulary: " + vocabulary.size());
		Iterator<String> myVocabIterator = vocabulary.iterator();
		
		println (" - Class iterator: " + myDictionary.keySet());
		int noDocsWithWordInClass = 0;
		int noDocsInClass = 0;
		
		while(myVocabIterator.hasNext()){			
			String word = myVocabIterator.next();
			//println("word: " + word);
			// Check if that word is in the docs of each class; if it is, increase the counter
			// This calculates number of docs containing that word in that class
			Map<String, Double> condProbPerClass = new HashMap<String, Double>();
			Iterator<String> myClassesIterator = myDictionary.keySet().iterator();
			while (myClassesIterator.hasNext()){
				
				noDocsWithWordInClass = 0;
				noDocsInClass = 0;
			
				String myClass = myClassesIterator.next();
				//println(" - class: " + myClass);
				
				List<String> myDocs = new ArrayList<>();
				myDocs = myDictionary.get(myClass);
				noDocsInClass = myDocs.size();
				//println(" - class: " + myClass + " with # docs: " + noDocsInClass);
				//println("docs in class for word " +word +" - " + noDocsInClass);
				// Count the number of docs with that word in that class
				
				Iterator<String> myDocIterator = myDocs.iterator();
				while (myDocIterator.hasNext()){
					String myDoc = myDocIterator.next();
					if (myDoc.contains(word)){
						noDocsWithWordInClass++;
					}
					
					
				}
				//println("docs with word in class "+noDocsWithWordInClass);
				// Now calculate the conditional probability for this class and word

				double condprob = new Double(((noDocsWithWordInClass + 1.0)/(noDocsInClass + 2.0)));
				condProbPerClass.put(myClass, condprob);
			}
			condProbab.put(word, condProbPerClass);
			//println("word: " + word + "  ---  prob: " + condProbPerClass);
		}
		//println("Conditional probability per word/class calculated.");
		//println(condProbab.toString().substring(0, 100));
	}

private static void calculateCondProbMultinomial() {
	println (" - Calculating multinomial cond prob with vocabulary: " + vocabulary.size());
	Iterator<String> myVocabIterator = vocabulary.iterator();
	
	println (" - Class iterator: " + myDictionary.keySet());
	int noDocsWithWordInClass = 0;
	int noDocsInClass = 0;
	
	while(myVocabIterator.hasNext()){	
		long totalVocab = vocabulary.size();
		String word = myVocabIterator.next();
		//println("word: " + word);
		// Check if that word is in the docs of each class; if it is, increase the counter
		// This calculates number of docs containing that word in that class
		Map<String, Double> condProbPerClass = new HashMap<String, Double>();
		Iterator<String> myClassesIterator = classVocabulary.keySet().iterator();
		while (myClassesIterator.hasNext()){
			
			noDocsWithWordInClass = 0;
			noDocsInClass = 0;
		
			String myClass = myClassesIterator.next();
			//println(" - class: " + myClass);
			
			List<String> myClassVocab = new ArrayList<String>();
			myClassVocab = classVocabulary.get(myClass);
			long classVocab = myClassVocab.size();
			
			long countTokens = myClassVocab.stream().filter(s -> s.equals(word)).count();
			
			
			//println("docs with word in class "+noDocsWithWordInClass);
			// Now calculate the conditional probability for this class and word

			double condprob = new Double((countTokens + 1.0)/(classVocab+totalVocab));
			condProbPerClass.put(myClass, condprob);
		}
		condProbab.put(word, condProbPerClass);
		//println("word: " + word + "  ---  prob: " + condProbPerClass);
	}
	//println("Conditional probability per word/class calculated.");
	//println(condProbab.toString().substring(0, 100));
}
	
	
	// Read document, filter lines starting with a word and return lines in a List
	public static List<String> readLines(String pathT, String filter){
		List<String> myLines = new ArrayList<>();
		Path path = Paths.get(pathT);
        try (Stream<String> lines = Files.lines(path)) {
        	lines.filter(str -> str.startsWith(filter)).forEach(s -> myLines.add(s));
        } catch (IOException ex) {
        	println(ex.getMessage());
        }
		return myLines;
	}
	
	// Read full document and return lines in a List
	private static List<String> readLines(String pathT){
		List<String> myLines = new ArrayList<>();
		Path path = Paths.get(pathT);
        try (Stream<String> lines = Files.lines(path)) {
    		if (setLowerCase){
    			lines.forEach(s -> myLines.add(s.toLowerCase()));
    		} else {
    			lines.forEach(s -> myLines.add(s));
    		}
        	
        } catch (IOException ex) {
        	println(ex.getMessage());
        }
		return myLines;
	}
	
	private static void writeOutput(String text){
		List<String> myLines = new ArrayList<>();
		try (BufferedWriter bw = new BufferedWriter(new FileWriter("/Users/JoseMa/output.txt"))){
			bw.write(text);
		} catch (IOException e){
			println(e.getMessage());
		}
	//	Files.write("/Users/JoseMa/output.txt", text.getBytes());
        
	}
	
	// Split list of documents in 2.
	private static List<List<String>> docSplit(int bags, List<String> docList){
		List<List<String>> docSplit = new ArrayList<List<String>>(bags);

		docSplit.add(new ArrayList<String>(docList.subList(0, bags-1)));
		docSplit.add(new ArrayList<String>(docList.subList(bags, docList.size()-1)));

		return docSplit;
	}
	 
	/*
	 * Split list of documents for k-fold cross validation.
	 * key 0 = test data
	 * key 1 = training data
	 */
		private static Map<Integer,List<String>> docSplit(int iter, int bags, List<String> docList){
			println("k-fold iteration from " +bags*(iter-1)+ " to " + (bags*iter-1));
			
			Map<Integer,List<String>> docSplit = new HashMap<Integer,List<String>>();
			List<String> docListTemp = new ArrayList<String>(docList);
			List<String> testSet = new ArrayList<String>(docListTemp.subList(bags*(iter-1), bags*iter));
			docSplit.put(0,testSet);
			docListTemp.removeAll(testSet);
			docSplit.put(1,docListTemp);

			return docSplit;
		}
	
	private static void println(String s){
		System.out.println(s);
	}
	
	private static void getClasses(List<String> textLines){
		
		
		for (int i=0; i<textLines.size(); i++){
			if (textLines.get(i).contains("\t")){
				classList.add(textLines.get(i).split("\t")[0]);
			}
		}

	}
	
	private static void printHelp(){
		println("*****************************");
		println("*      RUN PARAMETERS       *");
		println("*****************************");
		println(" ./java -jar Exercise2.jar <case sensitive> <remove dots or commas> <model type> <k-fold number of bags>");
		println(" <case sensitive>: 1 = yes, 0 = NO");
		println(" <remove dots or commas>: 1 = YES, 0 = no");
		println(" <model type>: BERNOULLI, multinomial");
		println(" <k-fold number of bags>: [5] - integer setting the number of bags to use in k-fold");
	}
	

}
