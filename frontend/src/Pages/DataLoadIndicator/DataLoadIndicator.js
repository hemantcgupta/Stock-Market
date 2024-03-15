import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import DataTableComponent from '../../Component/DataTableComponent';
import Access from '../../Constants/accessValidation';
import color from '../../Constants/color'
import $ from 'jquery';
import '../../App';
import './DataLoadIndicator.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import _ from 'lodash';

const apiServices = new APIServices();

class DataLoadIndicator extends Component {


  constructor(props) {
    super(props);
    this.state = {
      rowData: [],
      column: [],
      loading: false
    };
    this.sendEvent('1', 'viewed Data Load Indicator Page', 'dataLoadIndicator', 'Data Load Indicator');
  }


  sendEvent = (id, description, path, page_name) => {
    var eventData = {
      event_id: `${id}`,
      description: `User ${description}`,
      where_path: `/${path}`,
      page_name: `${page_name} Page`
    }
    eventApi.sendEvent(eventData)
  }

  componentDidMount() {
    if (Access.accessValidation('Data Load Indicator', 'dataLoadIndicator')) {
      this.getDataLoadIndicator();
    } else {
      this.props.history.push('/404')
    }
  }

  getDataLoadIndicator = () => {
    var self = this;
    self.setState({ loading: true })
    apiServices.getDataLoadIndicator().then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        self.setState({ rowData: rowData, column: columnName })
      }
    });
  }


  render() {
    const { loading } = this.state;

    return (
      <div className='DataLoadIndicator'>
        <TopMenuBar {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 DataLoadIndicator-main">
            <div className="navdesign">
              <div className="col-md-12 col-sm-12 col-xs-12 top-heading">
                <section>
                  <h2>{`Data Load Indicator`}</h2>
                </section>
              </div>

              <div className="x_content" style={{ marginTop: '15px', padding: '10px' }}>
                <DataTableComponent
                  rowData={this.state.rowData}
                  columnDefs={this.state.column}
                  loading={loading}
                  channel={true}
                />
              </div>
            </div>

          </div>
        </div>
      </div >

    );
  }
}

export default DataLoadIndicator;