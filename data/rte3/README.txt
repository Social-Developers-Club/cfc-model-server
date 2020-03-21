
*******************************************************
* A SEARCH TASK DATASET FOR GERMAN TEXTUAL ENTAILMENT *
*******************************************************

This is a Textual Entailment (TE) dataset from German social media data. 
It consists of a set of RTE3-alike Text/Hypothesis pairs compiled from 
a German self-help web forum for computer problems. Each Text corresponds 
to the first post in a thread, usually describing a problem to be 
diagnosed. The corresponding Hypotheses have been created with a sequence 
of crowd-sourcing tasks. The dataset contains 3014 T/H pairs in total and 
is split into a development and a test set with equal amounts of pairs.
Both sets contain 86 positive and 1421 negative entailment pairs,
respectively.


If you use this dataset, please cite the following paper:
B. Zeller and S. Padó: A Textual Entailment Dataset from German Web Forum 
Text. In Proceedings of the Tenth International Conference on Computational 
Semantics (IWCS 2013), Potsdam, Germany.

The dataset was created in the context of the EC-funded project 
EXCITEMENT (EXploring Customer Interactions through Textual EntailMENT): 
http://www.excitement-project.eu/


***************
* DATA FORMAT * 
***************

The following Document Type Defintion declares the dataset format:

<?xml version="1.0" encoding="UTF-8" ?>
<!ELEMENT entailment-corpus (pair+)>
<!ELEMENT pair (t,h)>
<!ATTLIST pair
          id CDATA #REQUIRED
	  entailment (ENTAILMENT|NONENTAILMENT) #REQUIRED
	  task (IR|IE|QA|SUM) #REQUIRED >
<!ELEMENT t (#PCDATA)>
<!ELEMENT h (#PCDATA)>
 

***********
* LICENSE *
***********

Creative Commons Attribution-ShareAlike (CC-BY-SA):

This work is licensed under the Creative Commons Attribution-ShareAlike 3.0 
Unported License. To view a copy of this license, visit 
http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to Creative 
Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.


***********
* CONTACT *
***********

For any questions or comments, please contact Britta Zeller: 
zeller AT cl.uni-heidelberg.de



---
Department of Computational Linguistics, Heidelberg University, 2013.
