document.addEventListener('DOMContentLoaded', function () {
    htmx.engine();
});

$(document).ready(function() {
    function showConfirmModal(message, image) {
        $('#class-name-modal').text(message);
        document.querySelector('#confirmModal img').src = `data:image/png;base64,${image}`
        document.getElementById('confirmModal').style.display = 'flex';
    }

    function showErrorModal(image) {
        document.querySelector('#errorModal img').src = `data:image/png;base64,${image}`
        document.getElementById('errorModal').style.display = 'flex';
    } 

    $('.modal-close-btn').on('click', function() {
        $('#file-form')[0].reset()
        $('.modal').hide();
    });

    $('.modal-confirm-btn').on('click', function() {
        $('.modal').hide();      
    });

    $('#file-form').submit(function (e) {
        var form_data = new FormData($('#file-form')[0]);

        $.ajax({
            type: "POST",
            url: "/process-data",
            data: form_data,
            processData: false,
            contentType: false,
            cache: false,
            timeout:5000,
            success: function (data) {
                if (data.index == '8') {
                    showErrorModal(data.image);
                } else {
                    showConfirmModal(data.message, data.image);
                }
            },
            error: function (xhr, status, error) {
                console.log(xhr)
                console.log(status)
                console.log(error)
                console.log("DEU ERRO AI")
            }
        });
        
        e.preventDefault();
    })

});