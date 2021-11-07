# -*- coding: utf-8 -*-

import pickle
from transformer_framework import Transformer
from transformer_framework import predict
from  nltk.translate.bleu_score import sentence_bleu
NUM_LAYERS = 2
D_MODEL = 512
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.2

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE=tokenizer.vocab_size+2
model =Transformer(vocab_size = VOCAB_SIZE, num_layers=NUM_LAYERS, units = UNITS, d_model = D_MODEL, num_heads = NUM_HEADS,dropout = DROPOUT)
model.load_weights('transformer')

def function1(sent):
    
      '''
      The function accepts a string as input and return the predicted sentence by the chatbot
      
      '''
        
      predicted_sent=predict(sent, model)
    
      return predicted_sent


def function2(sent,ref):
    
    '''
    The function takes the question asked by the user and the result expected by the user as argument and returns the bleu score based on the prediction made by the bot and the actual result.
   
    '''
    candidate=predict(sent,model).split()
    ref=ref.split()
  
    bleu_score=sentence_bleu(candidate,ref,weights=[0.25,0.25,0.25,0.25])
    return bleu_score



import streamlit as st
def main():
    st.title("CONVERSATIONAL AI")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sentence = st.text_input("INPUT","Type Here")
    
    result=""
    if st.button("function1"):
        result=function1(sentence)
    st.success('BOT-> {}'.format(result))
  

if __name__=='__main__':
    main()