import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent'
import { string } from '../../../../Constants/string'
import Spinners from "../../../../spinneranimation";

const apiServices = new APIServices();

class TopAgentOD extends Component {

  constructor(props) {
    super(props);
    this.state = {
      agentData: [],
      odData: [],
      column: [],
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    var self = this;
    const { gettingYear, gettingMonth, regionId, countryId, cityId, currency } = props;
    self.setState({ loading: true })
    apiServices.getTopAgentWithOD(gettingYear, gettingMonth, regionId, countryId, cityId, currency).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        self.setState({ agentData: rowData, column: columnName })
      }
    });
  }

  render() {
    return (
      <div>
        <div className="x_title" style={{ marginTop: '5px' }}>
          <h2>Top 10 Agents with Top 5 OD's by Incremental Revenue</h2>
          <div className="clearfix"></div>
        </div>
        {this.state.loading ? <Spinners /> : this.state.agentData.length > 0 ?
          <div className='agents-tables'>
            {this.state.agentData.map((element, i) =>
              <div>
                <div className="x_title" style={{ borderBottom: 'none' }}>
                  <h2>{`${i + 1}. ${element.AgentName}`}</h2>
                </div>
                <div className="x_content">
                  <DataTableComponent
                    rowData={element.Data}
                    columnDefs={this.state.column}
                    autoHeight={`autoHeight`}
                  />
                </div>
              </div>)}
          </div> : <h4 style={{ textAlign: 'center', marginTop: '10%' }}>No data to show</h4 >}
      </div>
    )
  }
}

export default TopAgentOD;