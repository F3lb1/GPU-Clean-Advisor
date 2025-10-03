def klass(photo):
  import tensorflow as tf
  from PIL import Image, ImageOps
  import numpy as np
  import warnings

  warnings.filterwarnings('ignore', category=UserWarning)


  np.set_printoptions(suppress=True)


  # Создаем кастомный слой для обхода проблемы с DepthwiseConv2D
  class SafeDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
      def __init__(self, *args, **kwargs):
          kwargs.pop('groups', None)  # Удаляем проблемный параметр
          super().__init__(*args, **kwargs)

  # Загружаем модель
  model = tf.keras.models.load_model(
      "keras_model.h5",
      custom_objects={'DepthwiseConv2D': SafeDepthwiseConv2D},
      compile=False
  )



  with open("labels.txt", "r") as f:
      class_names = [line.strip() for line in f.readlines()]


  # Обрабатываем изображение
  image = Image.open(photo).convert("RGB")
  image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
  image_array = np.asarray(image, dtype=np.float32)
  normalized_image = (image_array / 127.5) - 1
  data = np.expand_dims(normalized_image, axis=0)

  # Предсказание
  prediction = model.predict(data, verbose=0)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = float(prediction[0][index])

  # Вывод результатов
  result_class = class_name[2:].strip() if len(class_name) > 2 else class_name.strip()


  return(result_class)

import telebot

# Замени 'TOKEN' на токен твоего бота
bot = telebot.TeleBot("")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я твой Telegram бот. Напиши команду /hello, /bye, /pass, /emodji или /coin  ")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    if not message.photo:
        bot.reply_to(message, "Некоректнный формат")
    
    file_info = bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('/')[-1]
    downloaded_file = bot.download_file(file_info.file_path)
    with open(file_name, 'wb') as f:
        f.write(downloaded_file)
    bot.reply_to(message,klass(file_name))
# Запускаем бота
bot.polling()
