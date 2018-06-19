/* global $ */

function formatOutput (ajaxData, container) {
  let data = ajaxData.data

  if (data.length === 0) {
    console.log('Nothing here')
  } else {
    let outputHtml = ''

    for (var i = 0; i < data.length; i++) {
      let text = data[i].text
      let score = data[i].relevanceScore
      let lightness = (1 - score) * 100.0

      outputHtml += '<p class="response-line" style="color: hsl(140, 100%,' + lightness + '%)">' + text + '</p>'
    }
    container.append(outputHtml)
  }
}

$(document).ready(function () {
  $('#clear').on('click', function () {
    $('.text-input.main').val('')
    $('.response-container').empty()
  })

  $('#go').on('click', function () {
    let text = $('.text-input.main').val()
    let data = {body: text}

    $('.response-container').empty()

    $.ajax({
      url: 'http://localhost:8008/condense',
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
