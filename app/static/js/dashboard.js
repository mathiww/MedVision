$(document).ready(function() {
    function showConfirmModal(message) {
        $('#class-name-modal').text(message);
        document.getElementById('confirmModal').style.display = 'flex';
    }

    function showErrorModal() {
        document.getElementById('errorModal').style.display = 'flex';
    } 

    function hideModal() {
        $('.modal').hide();
    }

    $('.modal-close-btn').on('click', function() {
        $('#file-form')[0].reset()
        hideModal();
    });

    $('.modal-confirm-btn').on('click', function() {
        hideModal()        
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
            success: function (data) {
                console.log(data)
                if (data.index == '8') {
                    showErrorModal()
                } else {
                    showConfirmModal(data.message)
                }
            },
            error: function (xhr, status, error) {
                console.log("DEU ERRO AI")
            }
        });
        
        e.preventDefault();
    })

});