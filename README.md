# resnet_18-VS-resnet_50
data: image of CT and MRI , deep learning project , using pythorch library 
we will loader the data from local DB , split to three group train ,validation , test 
learning on train group and check in validation group , in the end we will check the result on test group 
part 1 comparing :
during the project i compare between to function 
1. CrossEntropyLoss ( number cant become negtive , when we fix the gradiant ) 
2. NLLLoss ( can be negtive is OK )

part 2 comparing :
we compare the result of two network learning 
1. resnet18 
2. resnet50 

need to know is the data is acctully not big so the maximum accuracy is 90%  resnet18 

you can check some Graf on wiki .


resnet_50 result :

  NLLloss function :

    validation set: Average loss: 0.454317, Accuracy: 112/138 (81%)

    test set: Average loss: 0.466939, Accuracy: 111/138 (80%)

  crossEntropyLoss : 
  
    validation set: Average loss: 0.393924, Accuracy: 113/138 (82%)

    test set: Average loss: 0.451591, Accuracy: 110/138 (80%)
    
 resnet_18:

NLLloss function :

	validation set: Average loss: -17.008425, Accuracy: 122/138 (88%)

	test set: Average loss: -17.043915, Accuracy: 124/138 (90%)
crossEntropyLoss:

	validation set: Average loss: 0.362180, Accuracy: 122/138 (88%)

	test set: Average loss: 0.379125, Accuracy: 118/138 (86%)



