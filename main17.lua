
-- Pedro Paulo Miranda de freitas
-- Marco de 2017
-- SMT UFRJ
-- Projeto DORIS
-- versao pra testar o dropout 

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'csvigo'
require 'optim'
require 'gnuplot'

----------------------------fucao qq que achei----------------------
function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end
----------------------------------------------------------------------

diretoriosReferencia = {

  -- the ones for training
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Backpack_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Backpack_POS2_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Backpack_POS3_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Coat_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Coat_POS2_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Coat_POS3_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Camera_Box_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Camera_Box_POS2_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Camera_Box_POS3_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Pink_Bottle_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Pink_Bottle_POS2_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Pink_Bottle_POS3_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Shoe_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Shoe_POS2_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Shoe_POS3_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/White_Jar_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/White_Jar_POS2_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Mult_Objs1_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Mult_Objs2_POS1_reference",


   --  the ones for testing
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Dark-Blue_Box_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Dark-Blue_Box_POS3_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Mult_Objs3_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Random_Background_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Towel_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Towel_POS2_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Towel_POS3_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Brown_Box_POS1_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Brown_Box_POS2_reference",
   "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Brown_Box_POS3_reference",


    }

  diretoriosAlvo={
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Backpack_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Backpack_POS2_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Backpack_POS3_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Coat_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Coat_POS2_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Black_Coat_POS3_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Camera_Box_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Camera_Box_POS2_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Camera_Box_POS3_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Pink_Bottle_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Pink_Bottle_POS2_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Pink_Bottle_POS3_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Shoe_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Shoe_POS2_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Shoe_POS3_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/White_Jar_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/White_Jar_POS2_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Mult_Objs1_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Mult_Objs2_POS1_target",

    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Dark-Blue_Box_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Dark-Blue_Box_POS3_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Mult_Objs3_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Random_Background_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Towel_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Towel_POS2_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Towel_POS3_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Brown_Box_POS1_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Brown_Box_POS2_target",
    "Reference_Object_frames_skip_17_full_aligned/EXTRA-Light/Brown_Box_POS3_target",
  }


local filesDiretorio = {}
local alvosDiretorio = {}
for dir = 1, #diretoriosAlvo do
  local alvos = {}
  local lfd = require 'lfs'
  local filesHere= 0
  dir_path = diretoriosAlvo[dir]
  for file in lfs.dir(dir_path) do
    filesHere = filesHere+1
    if file =='object_frames.txt' then
      local file = io.open(dir_path..'/object_frames.txt',"rb")
      while true do
         local var = file:read "*number"
        if not var then break
        else    local selecionados = {}

          alvos[#alvos+1] = var
        end
      end
      file:close()
    end
  end
-----------------aqui nos temos em alvos todos os intervalos que contem um objeto
  -- print(alvos[1])
  alvosDiretorio[#alvosDiretorio+1] = alvos
  filesDiretorio[#filesDiretorio+1] = filesHere -4
end

-- print(alvosDiretorio) ----limites
 --print(filesDiretorio)-----numero de arquivos


----------------------carregando paths e labels para diretorios treino 1-19---------------------------------------------
list_of_filenames_target = {}
list_of_filenames = {}
trainingLabels = {}
testLables = {}

for i=1, 19 do

  local selecionados = {}

  if i== 18 or i == 19 then--verificar se nao é um dos multi objetos

    --neste caso iremos carregar todos os quadros do diretorio de multiobjetos e depois carregar a mesma quantidade da pasta de backgrouds
    --
    for imagem=alvosDiretorio[i][1], alvosDiretorio[i][2] do

        list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[i].."/frame_"..imagem..".png")
        list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[i].."/frame_"..imagem..".png")
        trainingLabels[#trainingLabels+1] = 1
        -- print(list_of_filenames)
        selecionados[#selecionados+1] = imagem

    end


      --  neste caso temos que carregar uma quantidade equivalente de frames de backgroud

        for imagem=1, #selecionados do
            list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[23].."/frame_"..imagem..".png")
            list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[23].."/frame_"..imagem..".png")
           trainingLabels[#trainingLabels+1] = 0
        end

  else

    if alvosDiretorio[i][1] then
      for imagem=alvosDiretorio[i][1], alvosDiretorio[i][2] do

          list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[i].."/frame_"..imagem..".png")
          list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[i].."/frame_"..imagem..".png")
          trainingLabels[#trainingLabels+1] = 1
          -- print(list_of_filenames)
          selecionados[#selecionados+1] = imagem
      end
    end

    if alvosDiretorio[i][3] then
      for imagem=alvosDiretorio[i][3], alvosDiretorio[i][4] do
        list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[i].."/frame_"..imagem..".png")
        list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[i].."/frame_"..imagem..".png")
        trainingLabels[#trainingLabels+1] = 1
        selecionados[#selecionados+1] = imagem
      end
    end


    ---adicionando os frames que nao tem objeto na mesmo proporcao
    local numSelecionados = #selecionados
    local iter = 0
    local reDo



    while iter < numSelecionados do
      local guess =  math.random(0, filesDiretorio[i])
      reDo = 0

      for _, v in pairs(selecionados) do
        if v == guess then
          reDo = 1
        end
      end
      --print(reDo)

      if reDo==0 then
        list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[i].."/frame_"..guess..".png")
        list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[i].."/frame_"..guess..".png")
        trainingLabels[#trainingLabels+1] = 0
        iter = iter+1
        selecionados[#selecionados+1] = guess
      end
    end

  end--fim do else que separa os multi objetos


end

----------------------carregando paths e labels para diretorios treino 20 em diante---------------------------------------------
--22 23
for i=20, #diretoriosAlvo do
  local selecionados = {}


  if i==22 then--ja neste temos um multiobjetos
    for imagem=alvosDiretorio[i][1], alvosDiretorio[i][2] do

        list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[i].."/frame_"..imagem..".png")
        list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[i].."/frame_"..imagem..".png")
        --print(testLables)
        testLables[#testLables+1] = 1
        selecionados[#selecionados+1] = imagem

    end


    --  neste caso temos que carregar uma quantidade equivalente de frames de backgroud

      for imagem=1, #selecionados do
          list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[23].."/frame_"..imagem..".png")
          list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[23].."/frame_"..imagem..".png")
         testLables[#testLables+1] = 0
      end

  elseif i~=23 then --garantir que a nao iteração sob a pasta que so contem os backgrounds



    if alvosDiretorio[i][1] then
      for imagem=alvosDiretorio[i][1], alvosDiretorio[i][2] do
          list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[i].."/frame_"..imagem..".png")
          list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[i].."/frame_"..imagem..".png")
          testLables[#testLables+1] = 1
          selecionados[#selecionados+1] = imagem
      end
    end
    if alvosDiretorio[i][3] then
      for imagem=alvosDiretorio[i][3], alvosDiretorio[i][4] do
        list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[i].."/frame_"..imagem..".png")
        list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[i].."/frame_"..imagem..".png")
        testLables[#testLables+1] = 1
        selecionados[#selecionados+1] = imagem
      end
    end


    ---adicninado os frames que nao tem objeto na mesmo proporcao
    local numSelecionados = #selecionados
    local iter = 0
    while iter < numSelecionados do
      local guess =  math.random(0, filesDiretorio[i])
      local reDo = 0
      for _, v in pairs(selecionados) do
        if v == guess then
          reDo = 1
        end
      end
      if reDo==0 then
        list_of_filenames[#list_of_filenames+1] = (diretoriosReferencia[i].."/frame_"..guess..".png")
        list_of_filenames_target[#list_of_filenames_target+1] = (diretoriosAlvo[i].."/frame_"..guess..".png")
        testLables[#testLables+1] = 0
        iter = iter+1
        selecionados[#selecionados+1] = guess
      end
    end
  end -- fim do else



end

print(#trainingLabels)
print(#testLables)
testLables = torch.ByteTensor(testLables):cuda()
trainingLabels = torch.ByteTensor(trainingLabels):cuda()
-- print(testLables)
-- print(list_of_filenames)

local number_of_files = #list_of_filenames

-- for i=1, number_of_files do
-- print(list_of_filenames[i])
-- print(list_of_filenames_target[i])
-- end

-- print(number_of_files)
-----------------------------------agora temos os caminhos para todas as imagens salvo em list_of_filenames!!!------------




-------------------------------inicio do loop que percorre todos os mapas de feature--------------------------------------------------------------------------------
local teste = {}
local tp = {}
local tn = {}
for camada=1, 17 do

    print("camada atual:"..camada)

    -- Load the model
    model = torch.load("resnet-50.t7"):cuda()

    ----------------------------------Camada 17 -----------------------------------------------------
    -- Remove the fully connected layer
    assert(torch.type(model:get(#model.modules)) == 'nn.Linear')

    -- aqui temos o modelo completo
    model:remove(#model.modules)
    model:remove(#model.modules)
    model:remove(#model.modules)

    if camada == 17 then

      -- polling pra ficar com +/- 6k parametros
      -- mudar para average pooling
      model:add(nn.SpatialAveragePooling(7,7,7,7):cuda())


    -------------------------------Camada 16 ------------------------------------------------------
    elseif camada == 16 then

      model:get(8):remove(3)
      model:add(nn.SpatialAveragePooling(7,7,7,7):cuda())
      print(model)
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 15 ------------------------------------------------------
    elseif camada == 15 then

      model:get(8):remove(3)
      model:get(8):remove(2)
      model:add(nn.SpatialAveragePooling(7,7,7,7):cuda())
    ---------------------------------------------'---------------------------------------------------

    ---------------------------------Camada 14 ------------------------------------------------------
    elseif camada == 14 then

      model:get(8):remove(3)
      model:get(8):remove(2)
      model:get(8):remove(1)
      model:add(nn.SpatialAveragePooling(14,14,14,14):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 13 ------------------------------------------------------
    elseif camada == 13 then

      model:remove(8)
      model:get(7):remove(6)
      model:add(nn.SpatialAveragePooling(14,14,14,14):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 12 ------------------------------------------------------
    elseif camada == 12 then

      model:remove(8)
      model:get(7):remove(6)
      model:get(7):remove(5)
      model:add(nn.SpatialAveragePooling(14,14,14,14):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 11 ------------------------------------------------------
    elseif camada == 11 then

      model:remove(8)
      model:get(7):remove(6)
      model:get(7):remove(5)
      model:get(7):remove(4)
      model:add(nn.SpatialAveragePooling(14,14,14,14):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 10 ------------------------------------------------------
    elseif camada == 10 then

      model:remove(8)
      model:get(7):remove(6)
      model:get(7):remove(5)
      model:get(7):remove(4)
      model:get(7):remove(3)
      model:add(nn.SpatialAveragePooling(14,14,14,14):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 09 ------------------------------------------------------
    elseif camada == 9 then

      model:remove(8)
      model:get(7):remove(6)
      model:get(7):remove(5)
      model:get(7):remove(4)
      model:get(7):remove(3)
      model:get(7):remove(2)
      model:add(nn.SpatialAveragePooling(14,14,14,14):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 08 ------------------------------------------------------
    elseif camada == 8 then

      model:remove(8)
      model:remove(7)
      model:add(nn.SpatialAveragePooling(21,21,21,21):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 07 ------------------------------------------------------
    elseif camada == 7 then

      model:remove(8)
      model:remove(7)
      model:get(6):remove(4)
      model:add(nn.SpatialAveragePooling(21,21,21,21):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 06 ------------------------------------------------------
    elseif camada == 6 then

      model:remove(8)
      model:remove(7)
      model:get(6):remove(4)
      model:get(6):remove(3)
      model:add(nn.SpatialAveragePooling(21,21,21,21):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 05 ------------------------------------------------------
    elseif camada == 5 then

      model:remove(8)
      model:remove(7)
      model:get(6):remove(4)
      model:get(6):remove(3)
      model:get(6):remove(2)
      model:add(nn.SpatialAveragePooling(21,21,21,21):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 04 ------------------------------------------------------
    elseif camada == 4 then

      model:remove(8)
      model:remove(7)
      model:remove(6)
      model:add(nn.SpatialAveragePooling(28,28,28,28):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 03 ------------------------------------------------------
    elseif camada == 3 then

      model:remove(8)
      model:remove(7)
      model:remove(6)
      model:get(5):remove(3)
      model:add(nn.SpatialAveragePooling(28,28,28,28):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 02 ------------------------------------------------------
    elseif camada == 2 then

      model:remove(8)
      model:remove(7)
      model:remove(6)
      model:get(5):remove(3)
      model:get(5):remove(2)
      model:add(nn.SpatialAveragePooling(28,28,28,28):cuda())
    -------------------------------------------------------------------------------------------------

    ---------------------------------Camada 01 ------------------------------------------------------
    elseif camada == 1 then

      model:remove(8)
      model:remove(7)
      model:remove(6)
      model:remove(5)
      model:add(nn.SpatialAveragePooling(21,21,21,21):cuda())


    end
-------------------------------------------------------------------------Fim do for das camadas de feature----------------------------------------------------------------------------------

print(model)

-----------------------------------------Agora vamos rodar as imagens uma por uma e salvar as features resultantes num vetor, vulgo featuresContainerRef---------------------------------------
model:evaluate()



local batch_size = 1 ---vamos rodar apenas 1 imagem por iteracao
-- local numDiretorios = 1
local numFiles = 0

-- for numDiretorios=1, #diretoriosReferencia do  -- #v is the size of v for lists.
-- for numDiretorios=1, 1 do  -- #v is the size of v for lists.

  for numFiles=1,number_of_files, batch_size do
  -- for numFiles=1,3, batch_size do

      local img_batch = torch.FloatTensor(3, 360, 640) -- batch numbers are the 3 channels and size of transform

      img_name = list_of_filenames[numFiles]
      if img_name  ~= nil then
          local img = image.load(img_name, 3, 'float')
          -- img = image.scale(img,640,360)
          -- local img = image.load(img_name)
          img_batch = image.scale(img, 640,360)
          -- print(img_batch:size())
          -- image.window()
          -- if numFiles==1 then image.display(img_batch) end

      end
     -------------------------- Jogando efetivamente na rede para obter as features-----------------------------------------------
     local output = model:forward(img_batch:cuda())

     ----------------------------Juntando tudo em um umico vetor
     local vista = nn.View(-1):setNumInputDims(4):cuda()
     local outputNorm = vista:forward(output)
     outputSize = outputNorm:size(1)
     ------------------------------a essa altura ja temos nosso mapa de feature e basta concaternar num vetorzao-------------------

      if not featuresContainerRef then
         featuresContainerRef = torch.FloatTensor(number_of_files, outputNorm:size(1)):zero()
      end
      ------------------essa e a parte em que concatenamos tudo

      featuresContainerRef[{{numFiles},{} }]:copy(outputNorm)
  end

-- end
-- print(featuresContainerRef[{{2},{}}] )
--------------------------pronto! ja temos todas features de referencia no featuresContainerRef!------------------------------------------------------



-------------------------agora vamos fazer o foward das features de alvo e salvar em featuresContainerAlvo-----------------------------------------

-- for numDiretorios=1, #diretoriosAlvo do  -- #v is the size of v for lists.
-- for numDiretorios=1, 1 do  -- #v is the size of v for lists.

  for numFiles=1,number_of_files, batch_size do
  -- for numFiles=1,3, batch_size do

      local img_batch = torch.FloatTensor(3, 360, 640) -- batch numbers are the 3 channels and size of transform

      img_name = list_of_filenames_target[numFiles]
      if img_name  ~= nil then
          local img = image.load(img_name, 3, 'float')
          -- local img = image.load(img_name)
          img_batch = image.scale(img, 640,360)

      end
     -------------------------- Jogando efetivamente na rede para obter as features-----------------------------------------------
     local output = model:forward(img_batch:cuda())

     ----------------------------Juntando tudo em um umico vetor
     local vista = nn.View(-1):setNumInputDims(4):cuda()
     local outputNorm = vista:forward(output)
     outputSize = outputNorm:size(1)
     ------------------------------a essa altura ja temos nosso mapa de feature e basta concaternar num vetorzao-------------------

      if not featuresContainerAlvo then
         featuresContainerAlvo = torch.FloatTensor(number_of_files, outputNorm:size(1)):zero()

--( •_•)>⌐■-■ mexer aqui caso alterecao entre diff e cat         
	 featuresDif = torch.FloatTensor(number_of_files, outputNorm:size(1)):zero()
      end
      ------------------essa e a parte em que concatenamos tudo
      featuresContainerAlvo[{{numFiles},{} }]:copy(outputNorm)

  end

-- end

----------------------------agora precisamos calcular a diferenca entre essas features para alimentar nossa fully connected-----------------

 for i=1, number_of_files do
   featuresDif[{{i},{} }] = featuresContainerAlvo[{{i},{} }]:csub(featuresContainerRef[{{i},{} }])
--   featuresDif[{{i},{} }] = torch.cat(featuresContainerAlvo[{{i},{} }],featuresContainerRef[{{i},{} }])

 end


--featuresDif = featuresContainerAlvo - featuresContainerRef



------------------------------Daqui pra baixo é a parte da fully connected-----------------------------------------------------------------

-------------------------------------------------------------------------------------------------
 -- print(featuresDif[{{1},{}}])
-- print(featuresDif:size())
-- print(trainingLabels:size())
-- print(testLables:size())

torch.save('featuresDif.t7',featuresDif)

vetorzaoTreino = featuresDif[{{1,3018},{}}]:cuda()
vetorzaoTeste = featuresDif[{{3019,4664},{} }]:cuda()
--vetorzaoTeste = featuresDif[{{2502,2759},{} }]:cuda()
-- print(vetorzaoTeste:size())
-- print(vetorzaoTreino:size())
-- print(vetorzaoTreino[{{2000},{}}])
-- print(featuresFinal:size())
-- vetorzaoTeste = featuresFinal[{{1,57},{} }]:cuda()
-- vetorzaoTreino = featuresFinal[{{58,112},{} }]:cuda()



-- adam trainer setting
local opt = {}
opt.optimization = 'adam'
opt.batch_size = 256
opt.train_size = 0
opt.test_size = 0
opt.epochs = 600

local optimState =  {
    learningRate = 1e-3,
    learningRateDecay =1e-6
    -- momentum = opt.momentum,
    -- nesterov = true,
    -- dampening = 0.0,
    -- weightDecay = opt.weightDecay,
}
---loss function
criterion = nn.BCECriterion()
criterion:cuda()


--- Defining the  modelFinal
local modelFinal = nn.Sequential()

modelFinal:add(nn.Dropout(threshold))
--( •_•)>⌐■-■  mexer aqui caso aterar entre dif e cat
modelFinal:add(nn.Linear(outputSize,1))
modelFinal:add(nn.Sigmoid())
modelFinal:cuda()


-- adagrad trainer
local parameters, gradParameters = modelFinal:getParameters()
----------------------novo modelo----------------------------
-- function computeScore(output, target, nCrops)
--    if nCrops > 1 then
--       -- Sum over crops
--       output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
--          --:exp()
--          :sum(2):squeeze(2)
--    end
--
--    -- Coputes the top1 and top5 error rate
--    local batchSize = output:size(1)
--
--    local _ , predictions = output:float():topk(5, 2, true, true) -- descending
--
--    -- Find which predictions match the target
--    local correct = predictions:eq(
--       target:long():view(batchSize, 1):expandAs(predictions))
--
--    -- Top-1 score
--    local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)
--
--    -- Top-5 score, if there are at least 5 classes
--    local len = math.min(5, correct:size(2))
--    local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)
--
--    return top1 * 100, top5 * 100
-- end

-- local function feval()
--   return criterion.output, gradParameters
-- end

-- local counter = 0
-- local losses = {}
-- local predictions = {}
-- acuracia = {}
-- epochs = opt.epochs
-- local numTreino = vetorzaoTreino:size(1)
-- local batch_size = opt.batch_size
-- local iterations = epochs * math.ceil(numTreino/batch_size)
-- local timer = torch.Timer()
-- local dataTimer = torch.Timer()

-- for i=1, iterations do
--      local dataTime = dataTimer:time().real
--       -- if x ~= parameters then
--       --   parameters:copy(x)
--       -- end
--       local start_index = counter*opt.batch_size +1
--       local end_index = math.min(vetorzaoTreino:size(1), (counter + 1)*opt.batch_size + 1)

--       if end_index == vetorzaoTreino:size(1) then
--         counter = 0
--       else
--         counter = counter + 1
--       end

--       local batch_inputs =  vetorzaoTreino[{{start_index, end_index},{} }]
--       local batch_targets = trainingLabels[{{start_index, end_index}}]


--       local output = modelFinal:forward(batch_inputs):float()
--       local batchSize = output:size(1)
--       local loss = criterion:forward(modelFinal.output, batch_targets)

--       modelFinal:zeroGradParameters()
--       criterion:backward(modelFinal.output, batch_targets)
--       modelFinal:backward(batch_inputs, criterion.gradInput)

--       -- optim.sgd(feval, self.params, self.optimState)
--       optim.adam(feval, parameters, optimState)

--       --
--       -- local top1, top5 = computeScore(output, batch_targets, 1)
--       -- top1Sum = top1Sum + top1*batchSize
--       -- top5Sum = top5Sum + top5*batchSize
--       -- lossSum = lossSum + loss*batchSize
--       -- N = N + batchSize

--       -- print( dataTime, loss)

--       -- check that the storage didn't get changed due to an unfortunate getParameters call
--       -- assert(self.params:storage() == modelFinal:parameters()[1]:storage())

--       timer:reset()
--       dataTimer:reset()
-- end



-------------------------------------------------------------
local counter = 0
local losses = {}
local feval = function(x)

    if x ~= parameters then
      parameters:copy(x)
    end
    local start_index = counter*opt.batch_size +1
    local end_index = math.min(vetorzaoTreino:size(1), (counter + 1)*opt.batch_size + 1)

    if end_index == vetorzaoTreino:size(1) then
      counter = 0
    else
      counter = counter + 1
    end
    -- print(counter)
    local batch_inputs =  vetorzaoTreino[{{start_index, end_index},{} }]
    -- local batch_inputs =  vetorzao[{{start_index, end_index}, {}}]

    local batch_targets = trainingLabels[{{start_index, end_index}}]
    -- batch_targets:cuda()

  -- 1. compute outputs (log probabilities) for each data point
    local batch_outputs = modelFinal:forward(batch_inputs)

  -- 2. compute the loss of these outputs, measured against the true labels in batch_target
    local batch_loss = criterion:forward(batch_outputs, batch_targets)

    gradParameters:zero()

  -- 3. compute the derivative of the loss wrt the outputs of the modelFinal
    local dloss_doutput = criterion:backward(batch_outputs, batch_targets)

  -- 4. use gradients to update weights
     modelFinal:backward(batch_inputs, dloss_doutput)

  -- print(batch_loss:size())
     losses[#losses + 1] =batch_loss



  -- optim expects us to return
  -- loss, (gradient of loss with respect to the weights that we're optimizing)
  return batch_loss, gradParameters
end

  -- local losses = {}
  local predictions = {}
  acuracia = {}
  epochs = opt.epochs
  local numTreino = vetorzaoTreino:size(1)
  local batch_size = opt.batch_size
  -- print(numTreino)
  local iterations = epochs * math.ceil(numTreino/batch_size)

  for i=1, iterations do

      optim.adam(feval, parameters, optimState)
      --if numero de epocas == 200, salvar tp, tn, e acuracia em arquivo
      -- for iter=math.floor(iterations/6),iterations,math.floor(iterations/6)
  end


  torch.save('modelFinal.t7',modelFinal)

  --calculando fp fn tp tn acuracia

  local test_output = modelFinal:forward(vetorzaoTeste):float()
  test_output:round()
  -- print(test_output:size())

  dif = torch.eq(testLables:float(), test_output)
  sum = torch.sum(dif)
  numFramesTeste = 1646
  local percent = sum/numFramesTeste*100
  teste[#teste+1] = percent

  --true positive
  tp[#tp+1] = (torch.cmul(test_output, testLables:float()):sum()/(testLables:float():sum()))*100
  --true negative
  tn[#tn+1] = ((test_output+testLables:float()):eq(0):float():sum()/(numFramesTeste - testLables:float():sum()))*100
  print(teste)
  print(tp)
  print(tn)




--------------------limpando as variaveis globais-----------------------
      featuresContainerRef = nil
      featuresContainerAlvo = nil
      featuresDif = nil


end

print(teste)
gnuplot.pngfigure('acuracia.png')
gnuplot.plot({ 'acuracia',
  torch.range(1, #teste),
  torch.Tensor(teste),
  '-'})
gnuplot.xlabel('camada')
gnuplot.ylabel('acuracia')
gnuplot.plotflush()


gnuplot.pngfigure('tn.png')
gnuplot.plot(
             {'trueNegative', torch.range(1, #tn), torch.Tensor(tn), '-'})
gnuplot.xlabel('camada')
gnuplot.ylabel('acuracia')
gnuplot.plotflush()

gnuplot.pngfigure('tn.png')
gnuplot.plot(
             {'truePositive', torch.range(1, #tp), torch.Tensor(tn), '-'})
gnuplot.xlabel('camada')
gnuplot.ylabel('acuracia')
gnuplot.plotflush()

print(tp)
print(tn)


-- gnuplot.pngfigure('truePositive.png')
-- gnuplot.plot({'truePositive', torch.range(1, #tp), torch.Tensor(tn), '-'})
-- gnuplot.xlabel('camada')
-- gnuplot.ylabel('true Positive')
-- gnuplot.plotflush()
--
--
-- print(tn)
-- gnuplot.pngfigure('trueNegative.png')
-- gnuplot.plot({'trueNegative', torch.range(1, #tn), torch.Tensor(tn), '-'})
-- gnuplot.xlabel('camada')
-- gnuplot.ylabel('true Negative')
-- gnuplot.plotflush()

gnuplot.pngfigure('all.png')
gnuplot.raw(
            'set terminal pngcairo size 3840,2160\
            set lmargin at screen 80.0/600\
              set rmargin at screen 579.0/600')
gnuplot.plot(
           {'ac', torch.range(1, #teste), torch.Tensor(teste), '+-'},
           {'tn', torch.range(1, #tn), torch.Tensor(tn), '+-'},
         {'tp', torch.range(1, #tp), torch.Tensor(tp), '+-'})
gnuplot.xlabel('camada')
gnuplot.ylabel('skip 17')
gnuplot.plotflush()

gnuplot.pngfigure('truePositive.png')
gnuplot.plot(
             {'truePositive', torch.range(1, #tp), torch.Tensor(tn), '-'})
gnuplot.xlabel('camada')
--gnuplot.ylabel('')
gnuplot.plotflush()
