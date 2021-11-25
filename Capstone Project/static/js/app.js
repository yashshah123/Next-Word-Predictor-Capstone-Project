var data = []
var token = ""

jQuery(document).ready(function () {
    var slider = $('#max_words')
    slider.on('change mousemove', function (evt) {
        $('#label_max_words').text('Top k words: ' + slider.val())
    })

    $('#input_text').on('keyup', function (e) {
        if (e.key == ' ') {
            $.ajax({
                url: '/get_end_predictions',
                type: "post",
                contentType: "application/json",
                dataType: "json",
                data: JSON.stringify({
                    "input_text": $('#input_text').val(),
                    "top_k": slider.val(),
                }),
                beforeSend: function () {
                    $('.overlay').show()
                },
                complete: function () {
                    $('.overlay').hide()
                }
            }).done(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
                $('#text_bert').val(jsondata['bert'])
                $('#input_length').val(jsondata['input_length'])
                $('#pred_length').val(jsondata['predicted_words_length'])
                $('#pred_words').val(jsondata['predictied_words_used'])
                $('#accuracy').val(jsondata['accuracy'])
            }).fail(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
            });
        }
    })

})