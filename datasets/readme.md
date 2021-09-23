NSFNet datesets:
  1ms:
      training : train_nsf_200_200
      validation : eval_nsf_200_200
      test : nsf_test_data_per_200_1ms
  2ms:
      training : train_nsf_1KB_2ms_200
      validation : eval_nsf_1KB_2ms_200
      test : nsf_test_data_per_200_2ms
      
GBNFNet datesets:
  1ms:
      training : train_gbn_per_200_1KB_1ms_c350
      validation : eval_gbn_per_200_1KB_1ms_c350
      test : test_gbn_per_200_1KB_1ms_c350
  2ms:
      training : train_gbn_per_200_1KB_2ms
      validation : eval_gbn_per_200_1KB_2ms
      test : test_gbn_per_200_1KB_2ms
  
Mix datasets:
      training : mix
      validation : mix_eval
      test : 接使用NSF or GBN的test data測試就好
