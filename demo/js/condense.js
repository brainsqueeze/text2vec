/* global $ */

function formatOutput (ajaxData, container) {
  let data = ajaxData.data

  if (data.length === 0) {
    console.log('Nothing here')
  } else {
    let outputHtml = ''

    for (var i = 0; i < data.length; i++) {
      let text = data[i].text
      let score = data[i].score
      let lightness = score * 100.

      if (score > 0.5)
        outputHtml += `<p class="response-line" style="background-color: hsl(140, 100%, ${lightness}%)">${text}</p>`
      else
        outputHtml += `<p class="response-line" style="background-color: hsl(140, 100%, ${lightness}%); color: white">${text}</p>`
    }
    container.append(outputHtml)
  }
}

$(function () {
  $('#clear').on('click', function () {
    $('.text-input.main').val('')
    $('.response-container').empty()
  })

  $('#go').on('click', function () {
    let text = $('.text-input.main').val()
    let data = { text: text }

    $('.response-container').empty()

    $.ajax({
      url: '/summarize',
      data: data,
      type: 'POST',
      dataType: 'json',
      crossDomain: true,
      success: function (data, status, xhr) {
        let container = $('.response-container')
        formatOutput(data, container)
      },
      error: function (xhr, textStatus, errorThrown) {
        console.log(textStatus)
      }
    })
  })
})
